"""Reproducible experiment orchestration."""
import json
from pathlib import Path
import platform
import time

import cv2
import numpy as np

from .data import read_calibration,read_pfm
from .metrics import stereo_metrics
from .records import sha256_file,run_provenance
from .stereo import compute_sgbm,disparity_to_meters
from .cache import cache_key,save_predictions
from .models import load_image_adapter,infer_video_official
from .video import inspect_video,iter_rgb_frames
from .fusion import update
from .records import DepthPrediction


def fuse_sequence(frames,values,forward,backward,config):
    if len(frames)!=len(values) or len(forward)!=len(frames)-1 or len(backward)!=len(frames)-1:
        raise ValueError("Fusion frame/flow counts differ")
    state=None; outputs=[]; diagnostics=[]
    for index,(frame,depth) in enumerate(zip(frames,values)):
        prediction=DepthPrediction(depth,'relative','cached-DA-V2-Small','see-baseline-manifest','official','see-baseline-manifest')
        pair=(forward[index-1],backward[index-1]) if index else None
        result,state,debug=update(frame,prediction,state,config,flows=pair)
        outputs.append(result.values)
        diagnostics.append({'frame_id':frame.index,'reset':debug['reset'],'mean_weight':debug['mean_weight'],'valid_fraction':debug['valid_fraction']})
    return np.stack(outputs),diagnostics


def run_nyu(root: Path,output: Path) -> dict:
    from .data import load_nyu_split,load_nyu_sample
    from .metrics import evaluate_nyu,average_image_metrics

    root=Path(root); output=Path(output)
    provenance=run_provenance(root)
    split=load_nyu_split(root/'data/nyu/splits.mat')
    if len(split['development'])!=795 or len(split['test'])!=654:
        raise ValueError("Course NYU evaluation requires 795 development and 654 test IDs")
    if (output/'summary.json').exists():
        raise FileExistsError("Refusing to overwrite an existing NYU run")
    import torch
    import h5py

    torch.set_num_threads(4)
    model=load_image_adapter('unidepth',root)
    if not torch.cuda.is_available():
        raise RuntimeError("Full NYU run requires the verified CUDA environment")
    warm_rgb,_=load_nyu_sample(int(split['development'][0]),root/'data/nyu/nyu_depth_v2_labeled.mat')
    for _ in range(2): model.predict_rgb(warm_rgb)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    output.mkdir(parents=True,exist_ok=True)
    rows=[];latencies=[]
    with h5py.File(root/'data/nyu/nyu_depth_v2_labeled.mat','r') as dataset:
        if dataset['images'].shape!=(1449,3,640,480) or dataset['depths'].shape!=(1449,640,480):
            raise ValueError("NYU labeled MAT fields/shape differ from official data")
        for number,index in enumerate(split['test']):
            rgb=np.ascontiguousarray(dataset['images'][index].transpose(2,1,0))
            gt=dataset['depths'][index].T
            torch.cuda.synchronize();start=time.perf_counter()
            prediction=model.predict_rgb(rgb).values
            torch.cuda.synchronize();latency=time.perf_counter()-start
            latencies.append(latency)
            row={'image_index_zero_based':int(index),**evaluate_nyu(prediction,gt),'inference_s':latency}
            rows.append(row)
            np.savez_compressed(output/f'prediction_{int(index):04d}.npz',depth_m=prediction,image_index_zero_based=int(index))
            if (number+1)%50==0 or number==653:
                print(f'NYU {number+1}/654 | AbsRel {np.mean([r["abs_rel"] for r in rows]):.6f}',flush=True)
    summary={'metrics':average_image_metrics(rows),'per_image':rows,'protocol':json.loads((root/'configs/nyu-course.json').read_text()),'checkpoint_sha256':model.metadata['checkpoint_sha256'],'dataset_sha256':sha256_file(root/'data/nyu/nyu_depth_v2_labeled.mat'),'split_sha256':sha256_file(root/'data/nyu/splits.mat'),'timing':{'warmup_calls':2,'mean_image_s':float(np.mean(latencies)),'median_image_s':float(np.median(latencies)),'peak_allocated_mb':torch.cuda.max_memory_allocated()/1024**2,'peak_reserved_mb':torch.cuda.max_memory_reserved()/1024**2,'scope':'official infer(rgb), GPU synchronized; excludes file I/O/metrics/prediction compression'},'hardware':torch.cuda.get_device_name(),'torch_version':torch.__version__,'precision':model.metadata['precision'],'preprocessing':model.metadata['preprocessing_id'],'trained_or_fitted':'none'}
    summary['provenance']=provenance
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    (root/'results/nyu-baseline.json').write_text(json.dumps({key:value for key,value in summary.items() if key!='per_image'},indent=2,allow_nan=False),encoding='utf-8')
    return summary


def run_video_model(source: Path,output: Path,model_name: str,root: Path,limit: int=100,max_res: int=640,input_size: int=518,precision: str='fp16',adapter=None) -> dict:
    output=Path(output); root=Path(root)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing prediction cache: {output}")
    provenance=run_provenance(root)
    info=inspect_video(source,required_frames=limit)
    decoding_start=time.perf_counter()
    frames=list(iter_rgb_frames(source,limit))
    original_h,original_w=frames[0].rgb.shape[:2]
    scale=min(1,max_res/max(original_h,original_w))
    size=(round(original_w*scale),round(original_h*scale))
    for frame in frames:
        frame.rgb=cv2.resize(frame.rgb,size,interpolation=cv2.INTER_AREA)
    decode_s=time.perf_counter()-decoding_start
    torch=None
    if adapter is None:
        import torch
        torch.set_num_threads(4)
    device='cuda' if torch is not None and torch.cuda.is_available() else 'cpu'
    config={'model':model_name,'input_size':input_size,'max_res':max_res,'precision':precision,'frame_ids':[frame.index for frame in frames],'preprocessing':'official-model-specific; shared RGB resize INTER_AREA'}
    sources_path=root/'manifests/sources.json'
    sources=json.loads(sources_path.read_text()) if sources_path.is_file() else {}
    config['upstream_commit']=sources.get('video' if model_name=='vda' else 'framewise',{}).get('commit')
    stats={}
    if model_name=='vda':
        weights=root/'checkpoints/video_depth_anything_vits.pth'
        config['checkpoint_sha256']=sha256_file(weights)
        values=infer_video_official(np.stack([frame.rgb for frame in frames]),info.fps,root,input_size,precision,stats)
    else:
        loading_start=time.perf_counter()
        adapter=adapter or load_image_adapter(model_name,root,input_size,precision)
        config['checkpoint_sha256']=adapter.metadata['checkpoint_sha256']
        stats['load_s']=time.perf_counter()-loading_start
        # Warm-up is outside timed predictions and never changes the selected frames.
        for _ in range(2):
            adapter.predict_rgb(frames[0].rgb)
        if device=='cuda':
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        values=[]; latencies=[]
        for frame in frames:
            if device=='cuda': torch.cuda.synchronize()
            start=time.perf_counter(); prediction=adapter.predict_rgb(frame.rgb)
            if device=='cuda': torch.cuda.synchronize()
            latencies.append(time.perf_counter()-start)
            values.append(prediction.values)
        values=np.stack(values)
        stats.update(inference_s=sum(latencies),median_frame_s=float(np.median(latencies)),frame_latency_s=latencies,warmup_calls=2,peak_allocated_mb=torch.cuda.max_memory_allocated()/1024**2 if device=='cuda' else None,peak_reserved_mb=torch.cuda.max_memory_reserved()/1024**2 if device=='cuda' else None,device=device)
    metadata={'key':cache_key(info.source_sha256,config),'inference':config,'frame_ids':config['frame_ids'],'source_sha256':info.source_sha256,'source_decoded_frames':info.decoded_frames,'source_required_frames':limit,'source_compliant':info.compliant,'fps':info.fps,'kind':'relative','output_hw':list(values.shape[1:]),'decode_resize_s':decode_s,'timing':stats,'torch_version':torch.__version__ if torch is not None else None,'hardware':torch.cuda.get_device_name() if device=='cuda' else platform.processor(),'numpy_version':np.__version__,'opencv_version':cv2.__version__}
    # A caller's small smoke-test limit is independent of the course's fixed 100.
    metadata['requested_limit_satisfied']=info.decoded_frames>=limit
    metadata['course_required_frames']=100
    metadata['source_compliant']=info.decoded_frames>=100
    metadata['provenance']=provenance
    save_predictions(output,values,metadata)
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({key:metadata[key] for key in ['source_decoded_frames','source_compliant','output_hw','timing']},indent=2),flush=True)
    return metadata


def run_stereo(data_root: Path, output: Path, expected_pairs: int = 15) -> dict:
    data_root=Path(data_root); output=Path(output)
    pairs=sorted(p for p in data_root.glob('*') if p.is_dir() and (p/'im0.png').is_file())
    if len(pairs)!=expected_pairs:
        raise ValueError(f"Expected {expected_pairs} stereo pairs, found {len(pairs)} in {data_root}")
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for pair in pairs:
        calibration=read_calibration(pair/'calib.txt')
        left=cv2.imread(str(pair/'im0.png'),cv2.IMREAD_GRAYSCALE)
        right=cv2.imread(str(pair/'im1.png'),cv2.IMREAD_GRAYSCALE)
        if left is None or right is None:
            raise ValueError(f"Missing/corrupt stereo image in {pair}")
        disparities=int(np.ceil(calibration['ndisp']/16))*16
        start=time.perf_counter()
        prediction=compute_sgbm(left,right,num_disparities=disparities)
        elapsed=time.perf_counter()-start
        depth=disparity_to_meters(prediction,calibration['cam0'][0,0],calibration['baseline'],calibration['doffs'])
        np.savez_compressed(output/f'{pair.name}.npz',disparity_px=prediction,depth_m=depth.astype(np.float32),valid=np.isfinite(depth))
        row={'pair':pair.name,**stereo_metrics(prediction,read_pfm(pair/'disp0GT.pfm')),'inference_s':elapsed,'num_disparities':disparities,'calibration_sha256':sha256_file(pair/'calib.txt'),'gt_sha256':sha256_file(pair/'disp0GT.pfm')}
        rows.append(row)
        print(f"{pair.name}: MAE={row['mae_px']:.4f}px coverage={row['coverage']:.2%}",flush=True)
    summary={'pairs':rows,'mean_pair_mae_px':float(np.mean([row['mae_px'] for row in rows])),'mean_pair_coverage':float(np.mean([row['coverage'] for row in rows])),'pixel_weighted_coverage':sum(row['prediction_valid_count'] for row in rows)/sum(row['gt_valid_count'] for row in rows),'mean_inference_s':float(np.mean([row['inference_s'] for row in rows])),'invalid_prediction_rule':'nonfinite or disparity <= 0 px','aggregation':'arithmetic mean of per-pair MAE and coverage; counts also reported','configuration':{'num_disparities':'ceil(calib.ndisp / 16) * 16','block_size':5,'P1':200,'P2':800,'uniquenessRatio':10,'speckleWindowSize':100,'speckleRange':2,'disp12MaxDiff':1,'preFilterCap':31,'mode':'SGBM_3WAY'},'hardware':platform.platform(),'opencv_version':cv2.__version__,'model_training':'none','selection':'fixed independent defaults; no GT parameter tuning'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    return summary
