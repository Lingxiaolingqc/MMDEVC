"""Candidate-independent video evaluation and development-only selection."""
import csv
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .cache import cache_key,load_predictions,save_predictions
from .experiments import fuse_sequence
from .flow import estimate_pair
from .metrics import evaluate_video
from .records import Frame,sha256_file,run_provenance
from .video import iter_rgb_frames
from .warp import flow_consistency


EVALUATOR={'backend':'farneback','max_res':640,'pyr_scale':0.5,'levels':4,'winsize':21,'iterations':4,'poly_n':7,'poly_sigma':1.5,'flags':0}
CONSISTENCY={'alpha':0.01,'beta':0.5}


def fusion_flows(frames,reference_forward,reference_backward,requested):
    canonical={**EVALUATOR,**requested}
    if canonical==EVALUATOR:
        return reference_forward,reference_backward
    forward=[];backward=[]
    for previous,current in zip(frames,frames[1:]):
        f,b=estimate_pair(previous.rgb,current.rgb,requested)
        forward.append(f);backward.append(b)
    return np.stack(forward),np.stack(backward)


def prepare_evaluation(source,baseline_cache,output):
    output=Path(output)
    values,metadata=load_predictions(baseline_cache)
    if sha256_file(source)!=metadata['source_sha256']:
        raise ValueError("Evaluation source differs from prediction cache source")
    frames=list(iter_rgb_frames(source,limit=len(values)))
    if [frame.index for frame in frames]!=metadata['frame_ids']:
        raise ValueError("Evaluation source frame IDs/count differ from cache")
    height,width=values.shape[1:]
    for frame in frames:
        frame.rgb=cv2.resize(frame.rgb,(width,height),interpolation=cv2.INTER_AREA)
    output.mkdir(parents=True,exist_ok=True)
    key=cache_key(metadata['source_sha256'],{'flow':EVALUATOR,'consistency':CONSISTENCY,'frame_ids':metadata['frame_ids'],'output_hw':[height,width]})
    path=output/'fixed-flow.npz'
    start=time.perf_counter()
    if path.exists():
        with np.load(path,allow_pickle=False) as stored:
            if str(stored['key'])!=key:
                raise ValueError("Evaluation flow cache configuration mismatch")
            forward=stored['forward']; backward=stored['backward']; masks=stored['masks']
        mode='cached'
    else:
        forward=[];backward=[];masks=[]
        for previous,current in zip(frames,frames[1:]):
            f,b=estimate_pair(previous.rgb,current.rgb,EVALUATOR)
            forward.append(f);backward.append(b);masks.append(flow_consistency(f,b,CONSISTENCY))
        forward=np.stack(forward);backward=np.stack(backward);masks=np.stack(masks)
        np.savez(path,forward=forward,backward=backward,masks=masks,key=key)
        mode='computed'
    reference=max(float(np.percentile(values,75)-np.percentile(values,25)),1e-6)
    evaluator={'key':key,'flow':EVALUATOR,'mask_consistency':CONSISTENCY,'reference_scale':reference,'support_fraction':float(masks.mean()),'preparation_mode':mode,'preparation_s':time.perf_counter()-start,'limitations':'fixed Farneback evaluator shares a flow family with the fusion; scores are temporal proxies, not depth accuracy'}
    (output/'evaluator.json').write_text(json.dumps(evaluator,indent=2),encoding='utf-8')
    return frames,values,metadata,forward,backward,masks,evaluator


def _gradient(depth):
    dx=np.diff(depth,axis=2,append=depth[:,:,-1:])
    dy=np.diff(depth,axis=1,append=depth[:,-1:,:])
    return np.sqrt(dx*dx+dy*dy)


def edge_retention(candidate,baseline):
    reference=_gradient(baseline)
    edges=reference>max(float(np.percentile(reference,90)),1e-6)
    if not edges.any(): return 1.0
    return float(np.mean(_gradient(candidate)[edges])/np.mean(reference[edges]))


def synthetic_controls(config):
    rng=np.random.default_rng(42)
    height,width,count=48,64,10
    background=rng.integers(0,256,(height,width,3),np.uint8)
    texture=rng.integers(0,256,(24,12,3),np.uint8)
    frames=[];truth=[];object_masks=[]
    for index in range(count):
        mask=np.zeros((height,width),bool);mask[12:36,12+index:24+index]=True
        rgb=background.copy();rgb[12:36,12+index:24+index]=texture
        depth=np.full((height,width),2,np.float32);depth[mask]=8
        frames.append(Frame(index,index/24,rgb));truth.append(depth);object_masks.append(mask)
    truth=np.stack(truth)
    noisy=(truth+rng.normal(0,0.3,truth.shape)).astype(np.float32)
    forward=[];backward=[]
    for index in range(count-1):
        f=np.zeros((height,width,2),np.float32);f[object_masks[index],0]=1
        b=np.zeros_like(f);b[object_masks[index+1],0]=-1
        forward.append(f);backward.append(b)
    result,_=fuse_sequence(frames,noisy,np.stack(forward),np.stack(backward),config)
    edges=np.stack([cv2.dilate(mask.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)^cv2.erode(mask.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool) for mask in object_masks])
    newly_revealed=np.stack([object_masks[i-1]&~object_masks[i] for i in range(1,count)])
    step_frames=[Frame(i,i/24,np.zeros((height,width,3),np.uint8)) for i in range(15)]
    step_truth=np.array([2]*3+[10]*12,np.float32)[:,None,None]*np.ones((15,height,width),np.float32)
    zeros=np.zeros((14,height,width,2),np.float32)
    step_output,_=fuse_sequence(step_frames,step_truth,zeros,zeros,config)
    response=np.mean(step_output[3:],axis=(1,2))
    reaching=np.flatnonzero(response>=9.2)
    settling=int(reaching[0]+1) if len(reaching) else 13
    return {'synthetic_mse':float(np.mean((result-truth)**2)),'synthetic_baseline_mse':float(np.mean((noisy-truth)**2)),'synthetic_edge_mse':float(np.mean((result[edges]-truth[edges])**2)),'synthetic_disocclusion_mse':float(np.mean((result[1:][newly_revealed]-truth[1:][newly_revealed])**2)),'synthetic_settling_frames':settling}


def select_configuration(rows):
    eligible=[row for row in rows if row['name']!='baseline' and row['temporal'] is not None and row['edge_retention']>=0.90 and row['synthetic_settling_frames']<=2 and row.get('synthetic_mse',0)<=1.05*row.get('synthetic_baseline_mse',1)]
    if not eligible:
        raise ValueError("No improvement passed the declared edge/lag/synthetic-accuracy constraints")
    return min(eligible,key=lambda row:(row['temporal'],row['config'].get('history_cap',0)))


def development_configs():
    common={'flow':{'backend':'farneback','max_res':640},'consistency':CONSISTENCY,'scene_cut_rgb_difference':0.45,'photo_sigma':0.1,'relative_affine_alignment':False}
    configs=[('baseline',{**common,'method':'baseline','history_cap':0}),('ema',{**common,'method':'ema','history_cap':0.5}),('flow',{**common,'method':'flow','history_cap':0.5}),('occlusion',{**common,'method':'occlusion','history_cap':0.5})]
    for cap in [0.25,0.5,0.75]:
        for beta in [0.5,1.5]:
            for depth_sigma in [0.2,0.5]:
                configs.append((f'adaptive_w{cap}_b{beta}_d{depth_sigma}',{**common,'method':'adaptive','history_cap':cap,'consistency':{'alpha':0.01,'beta':beta},'depth_sigma':depth_sigma}))
    return configs


def run_development_sweep(source,baseline_cache,output):
    output=Path(output)
    frames,baseline,metadata,forward,backward,masks,evaluator=prepare_evaluation(source,baseline_cache,output)
    rows=[]
    for name,config in development_configs():
        start=time.perf_counter()
        candidate,diagnostics=fuse_sequence(frames,baseline,forward,backward,config)
        replay_s=time.perf_counter()-start
        metric=evaluate_video(candidate,backward,masks,evaluator['reference_scale'],metadata['frame_ids'])
        row={'name':name,'config':config,'temporal':metric['aligned_mae_normalized'],'edge_retention':edge_retention(candidate,baseline),'deviation_normalized':float(np.mean(np.abs(candidate-baseline))/evaluator['reference_scale']),'cached_replay_s':replay_s,**synthetic_controls(config),'evaluation':metric}
        rows.append(row)
        print(f"DEV {name}: temporal={row['temporal']:.6f} edge={row['edge_retention']:.3f} lag={row['synthetic_settling_frames']} synthMSE={row['synthetic_mse']:.5f}",flush=True)
    winner=select_configuration(rows)
    results={'source_sha256':metadata['source_sha256'],'frame_ids':metadata['frame_ids'],'evaluator':evaluator,'selection_policy':{'minimum_edge_retention':0.90,'maximum_step_settling_frames':2,'maximum_synthetic_mse_ratio':1.05,'objective':'lowest development temporal proxy among eligible non-baselines'},'rows':rows,'selected':winner}
    (output/'sweep.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')
    return results


def run_final_evaluation(source,baseline_cache,video_cache,config_path,output):
    from .application import background_blur
    from .render import compute_display_range,render_comparison,colorize

    output=Path(output)
    if (output/'summary.json').exists():
        raise FileExistsError("Refusing to overwrite an existing held-out evaluation")
    provenance=run_provenance(Path(config_path).resolve().parent.parent)
    config=json.loads(Path(config_path).read_text())
    if config.get('selection',{}).get('status') not in {'frozen before rollercoaster fusion evaluation','frozen on development before secondary engineering evaluation'}:
        raise ValueError("Final evaluation requires the frozen development-selected configuration")
    frames,baseline,metadata,forward,backward,masks,evaluator=prepare_evaluation(source,baseline_cache,output/'evaluation')
    candidate_forward,candidate_backward=fusion_flows(frames,forward,backward,config.get('flow',{}))
    reference,reference_metadata=load_predictions(video_cache,expected_frame_ids=metadata['frame_ids'])
    if reference_metadata['source_sha256']!=metadata['source_sha256'] or reference.shape!=baseline.shape:
        raise ValueError("Video-reference input/frames/output shape mismatch")
    output.mkdir(parents=True,exist_ok=True)
    configurations=[('baseline',{**config,'method':'baseline','history_cap':0}),('ema',{**config,'method':'ema'}),('flow',{**config,'method':'flow'}),('occlusion',{**config,'method':'occlusion'}),('adaptive',config)]
    rows=[];selected=None
    for name,candidate_config in configurations:
        start=time.perf_counter()
        values,diagnostics=fuse_sequence(frames,baseline,candidate_forward,candidate_backward,candidate_config)
        replay_s=time.perf_counter()-start
        metric=evaluate_video(values,backward,masks,evaluator['reference_scale'],metadata['frame_ids'])
        row={'method':name,'config':candidate_config,'temporal':metric['aligned_mae_normalized'],'edge_retention':edge_retention(values,baseline),'cached_replay_s':replay_s,'evaluation':metric,'diagnostics':diagnostics,**synthetic_controls(candidate_config)}
        rows.append(row)
        save_predictions(output/f'{name}.npz',values,{**metadata,'key':cache_key(metadata['key'],candidate_config),'fusion_config':candidate_config,'parent_baseline_key':metadata['key'],'configuration_sha256':sha256_file(config_path)})
        if name=='adaptive': selected=values
        print(f"TEST {name}: temporal={row['temporal']:.6f} edge={row['edge_retention']:.3f}",flush=True)
    display_range=compute_display_range(baseline)
    reference_range=compute_display_range(reference)
    rgb=np.stack([frame.rgb for frame in frames])
    methods={'RGB':rgb,'DA V2 Small (relative)':baseline,'VDA Small (relative)':reference,'Flow-guided fusion (relative)':selected}
    ranges={'DA V2 Small (relative)':display_range,'VDA Small (relative)':reference_range,'Flow-guided fusion (relative)':display_range}
    render_comparison(metadata['frame_ids'],methods,ranges,metadata['fps'],output/'comparison.mp4')
    app_config={'display_range':display_range,'focus_threshold':0.6,'transition':0.1,'blur_sigma':8}
    original_blur=np.stack([background_blur(frame.rgb,depth,app_config) for frame,depth in zip(frames,baseline)])
    fused_blur=np.stack([background_blur(frame.rgb,depth,app_config) for frame,depth in zip(frames,selected)])
    render_comparison(metadata['frame_ids'],{'Original-depth blur':original_blur,'Fused-depth blur':fused_blur},{},metadata['fps'],output/'background-blur.mp4')
    first=float(rows[0]['temporal']);last=float(rows[-1]['temporal'])
    summary={'source_sha256':metadata['source_sha256'],'source_frame_count':metadata['source_decoded_frames'],'requested_frames':metadata['source_required_frames'],'processed_frames':len(frames),'compliant_100_frames':metadata['source_compliant'],'fps':metadata['fps'],'output_hw':metadata['output_hw'],'configuration_sha256':sha256_file(config_path),'baseline_cache_key':metadata['key'],'display_ranges':ranges,'baseline_timing':metadata['timing'],'video_reference':reference_metadata,'evaluator':evaluator,'fusion_flow_config':config.get('flow',{}),'evaluation_stage':config.get('stage','primary evaluation of frozen development selection'),'rows':rows,'relative_temporal_proxy_reduction_percent':100*(first-last)/first,'interpretation':'Temporal proxy reduction is not metric-depth accuracy. Configuration selected on development data only; see evaluation_stage for primary versus engineering follow-up. All ablations share one fixed cap except unchanged baseline.','application_config':app_config,'raw_prediction_files':[str(Path(baseline_cache).resolve()),str(Path(video_cache).resolve()),str((output/'adaptive.npz').resolve())]}
    summary['provenance']=provenance
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    (output/'COMPLIANCE.txt').write_text('The official repository source contains 70 real frames, despite the instruction requesting 100. No frames were fabricated. This demo remains provisional pending a longer course source or TA clarification.\n' if not metadata['source_compliant'] else 'The required 100 genuine frames were processed.\n',encoding='utf-8')
    return summary
