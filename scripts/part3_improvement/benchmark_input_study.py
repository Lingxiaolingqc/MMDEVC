"""Measure isolated batch1 RGB-to-CPU-depth latency for frozen input variants."""
from pathlib import Path
import argparse,gc,json,sys,time
import h5py,numpy as np,torch,yaml
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from p3_depth.models.runtime import ImageModel
from p3_depth.residual.deployment import load_refiner as parent_refiner,image_inputs,predict_arrays,needs_flip,needs_edges
from p3_depth.residual_inputs.model import make_model
from p3_depth.residual_inputs.features import StageCapture
from p3_depth.residual_inputs.data import observable_inputs
from p3_depth.residual.model import corrected_depth
from p3_depth.residual.records import write_json,sha,now
from p3_depth.residual_inputs.common import snapshot
from p3_depth.residual.locking import study_lock


def load_refiner(path):
    saved=torch.load(path,map_location='cpu',weights_only=False);config=saved['config'];cfg=config['candidate']
    if 'branches' not in cfg:return parent_refiner(path)
    states=saved['model'];dummy={}
    if len(cfg['branches'])==1:
        channels=states['hidden.0.weight'].shape[1]-16
        dummy[cfg['branches'][0]]=np.empty((0,states['projection.weight'].shape[1],1,1))
    else:
        channels=states['body.hidden.0.weight'].shape[1]-16
        dummy={k:np.empty((0,states['projections.'+k+'.weight'].shape[1],1,1)) for k in cfg['branches']}
    model=make_model(channels,cfg,dummy).cuda().eval();model.load_state_dict(states);return model,config

@torch.inference_mode()
def live_prediction(rgb,adapter,refiner,config,edge_calibration):
    output=adapter.model.infer(torch.from_numpy(rgb.copy()).permute(2,0,1))
    original=output['depth'][0,0].float().cpu().numpy()
    if refiner is None:return original
    cfg=config['candidate']
    if 'branches' not in cfg:
        assert not needs_flip(cfg)
        arrays=image_inputs(rgb,original,cfg,config['feature_hw'],config['calibration'],internal=output['depth_features'][0].float().cpu().numpy())
        return predict_arrays(refiner,config,*arrays)[0].float().cpu().numpy()
    confidence=output['confidence'][0,0].float().cpu().numpy() if 'confidence' in cfg['extra_inputs'] else None
    features=observable_inputs(rgb,original,confidence,cfg,config['feature_hw'],edge_calibration)
    native={'init':output['depth_features']}
    if hasattr(adapter,'capture'):native.update(adapter.capture.values)
    # Match the saved native FP16 cache and measured CPU/GPU transfer contract.
    latents={k:torch.from_numpy(native[k].float().cpu().numpy().astype(np.float16)).float().cuda() for k in cfg['branches']}
    with torch.autocast('cuda',dtype=torch.float16):residual=refiner(torch.from_numpy(features[None]).cuda(),latents)
    return corrected_depth(torch.from_numpy(original[None,None]).cuda(),residual,'additive')[0,0].cpu().numpy()

@torch.inference_mode()
def cached_prediction(root,study,index,rgb,refiner,config,edge_calibration):
    with h5py.File(root/study['parent_output']/'cache/development/predictions.h5','r') as source:
        group=source[str(index)];original=group['original'][:]
        if refiner is None:return original
        cfg=config['candidate']
        if 'branches' not in cfg:
            arrays=image_inputs(rgb,original,cfg,config['feature_hw'],config['calibration'],internal=group['internal'][:])
            return predict_arrays(refiner,config,*arrays)[0].float().cpu().numpy()
        features=observable_inputs(rgb,original,group['confidence'][:],cfg,config['feature_hw'],edge_calibration)
    stages=root/study['output']/'stage_cache/development';meta=json.loads((stages/'metadata.json').read_text());pos=meta['spec']['ids'].index(index)
    arrays={k:np.load(stages/(k+'.npy'),mmap_mode='r') for k in cfg['branches']}
    latents={k:torch.from_numpy(np.array(v[pos:pos+1],copy=True)).float().cuda() for k,v in arrays.items()}
    with torch.autocast('cuda',dtype=torch.float16):residual=refiner(torch.from_numpy(features[None]).cuda(),latents)
    return corrected_depth(torch.from_numpy(original[None,None]).cuda(),residual,'additive')[0,0].cpu().numpy()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/residual_inputs_study.json')
    parser.add_argument('--checkpoints',nargs='+',required=True)
    parser.add_argument('--tag',default='selected_methods')
    args=parser.parse_args();study=json.loads((ROOT/args.config).read_text(encoding='utf-8'))
    if not args.tag.replace('_','').replace('-','').isalnum():raise ValueError('Simple benchmark tag required')
    paths=[(ROOT/path).resolve() for path in args.checkpoints]
    labels=['B0']+[path.parent.name for path in paths]
    if len(set(labels))!=len(labels):raise ValueError('Duplicate benchmark labels')
    directory=ROOT/study['output']/'benchmarks'/args.tag
    with study_lock(ROOT/study['output']):
        if directory.exists():raise FileExistsError('Use a new tag to preserve prior measurements')
        directory.mkdir(parents=True);snapshot(ROOT,directory)
        split=json.loads((ROOT/study['split']).read_text(encoding='utf-8'))
        ids=split['fit_ids'][:5]
        if sha(ROOT/study['data'])!=split['dataset_sha256']:raise ValueError('Dataset changed')
        with h5py.File(ROOT/study['data'],'r') as source:
            images=[source['images'][index].transpose(2,1,0).copy() for index in ids]
        settings=yaml.safe_load((ROOT/'configs/paths.local.yaml').read_text(encoding='utf-8'))
        adapter=ImageModel('unidepth',ROOT,settings,'cuda',precision='fp16',threads=4)
        adapter.model.requires_grad_(False)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        spec={'batch_size':1,'image_ids':ids,'warmups_per_block':5,'measured_calls_per_block':50,
              'blocks':2,'orders':[labels,labels[::-1]],'checkpoint_sha256':{str(path.relative_to(ROOT)):sha(path) for path in paths},
              'scope':'decoded RGB resident in CPU memory to full-resolution float32 CPU depth; includes foundation, required native-stage capture, CPU feature preparation/transfers and refiner; excludes file IO, visualization and model loading',
              'foundation':adapter.metadata,'selection':'first five fitting IDs; supplied checkpoints; no GT values used'}
        write_json(directory/'spec.json',spec);records={label:[] for label in labels};parameters={};parity={}
        for block,order in enumerate([list(range(len(labels))),list(range(len(labels)))[::-1]]):
            for index in order:
                label=labels[index];refiner=config=edge_cal=None
                if index:
                    refiner,config=load_refiner(paths[index-1])
                    parameters[label]=sum(p.numel() for p in refiner.parameters())
                    if needs_edges(config['candidate']):
                        edge_cal=json.loads((ROOT/study['parent_output']/'edge_features/metadata.json').read_text(encoding='utf8'))['calibration']
                else:parameters[label]=0
                if config is not None and set(config['candidate'].get('branches',[]))-{'init'}:
                    adapter.capture=StageCapture(adapter.model,[k for k in config['candidate']['branches'] if k!='init'])
                if block==0:
                    live=live_prediction(images[0],adapter,refiner,config,edge_cal)
                    cached=cached_prediction(ROOT,study,ids[0],images[0],refiner,config,edge_cal)
                    difference=float(np.max(np.abs(live-cached)));assert difference<=1e-5,('Live/cache parity failed',label,difference)
                    parity[label]=difference;write_json(directory/'live_cache_parity.json',parity)
                for image in images:live_prediction(image,adapter,refiner,config,edge_cal)
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
                for repeat in range(50):
                    image=images[repeat%len(images)]
                    torch.cuda.synchronize();start=time.perf_counter()
                    prediction=live_prediction(image,adapter,refiner,config,edge_cal)
                    torch.cuda.synchronize();times.append(time.perf_counter()-start)
                    if prediction.shape!=(480,640) or not np.isfinite(prediction).all():raise ValueError('Invalid benchmark output')
                row={'block':block+1,'seconds':times,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20}
                records[label].append(row);write_json(directory/'measurements.json',records)
                print(f'BENCH {label} block{block+1}: {np.mean(times)*1000:.3f} ms',flush=True)
                if hasattr(adapter,'capture'):adapter.capture.close();del adapter.capture
                del refiner,config;gc.collect();torch.cuda.empty_cache()
        summary={}
        for label,blocks in records.items():
            values=np.array([x for block in blocks for x in block['seconds']])
            summary[label]={'calls':len(values),'mean_ms':float(values.mean()*1000),'median_ms':float(np.median(values)*1000),
                            'p95_ms':float(np.quantile(values,.95)*1000),'fps_inverse_mean':float(1/values.mean()),
                            'peak_allocated_mib':max(b['peak_allocated_mib'] for b in blocks),'refiner_parameters':parameters[label],
                            'block_mean_ms':[float(np.mean(b['seconds'])*1000) for b in blocks]}
        write_json(directory/'result.json',{'status':'complete','at':now(),'spec':spec,'summary':summary,'live_cache_max_difference':parity,
                   'limitations':'Two opposite orders reduce but do not remove laptop thermal/background-load variation; short local benchmark, not universal hardware speed.'})
        print('PASS isolated batch1 benchmark: '+str(directory),flush=True)


if __name__=='__main__':main()
