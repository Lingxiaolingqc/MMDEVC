"""Measure batch1 RGB-to-CPU-depth latency, including required flip and refiner."""
from pathlib import Path
import argparse,gc,json,sys,time
import h5py,numpy as np,torch,yaml
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from p3_depth.models.runtime import ImageModel
from p3_depth.residual.deployment import load_refiner,image_inputs,predict_arrays,needs_flip,needs_edges
from p3_depth.residual.records import write_json,sha,snapshot,now
from p3_depth.residual.locking import study_lock


@torch.inference_mode()
def live_prediction(rgb,adapter,refiner,config,edge_calibration):
    tensor=torch.from_numpy(rgb.copy()).permute(2,0,1)
    output=adapter.model.infer(tensor)
    original=output['depth'][0,0].float().cpu().numpy()
    if refiner is None:return original
    cfg=config['candidate'];flipped=None
    if needs_flip(cfg):
        flipped=adapter.model.infer(tensor.flip(-1))['depth'][0,0].flip(-1).float().cpu().numpy()
    confidence=None
    if cfg['inputs'] in ('confidence','legacy') or 'confidence' in cfg.get('extra_inputs',[]):
        confidence=output['confidence'][0,0].float().cpu().numpy()
    latent=output['depth_features'][0].float().cpu().numpy() if cfg['inputs']=='frozen_features' else None
    arrays=image_inputs(rgb,original,cfg,config['feature_hw'],config['calibration'],
                        flipped=flipped,confidence=confidence,internal=latent,edge_calibration=edge_calibration)
    return predict_arrays(refiner,config,*arrays)[0].float().cpu().numpy()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/residual_study.json')
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
              'scope':'decoded RGB resident in CPU memory to full-resolution float32 CPU depth; includes foundation, required flip, CPU feature preparation/transfers and refiner; excludes file IO, visualization and model loading',
              'foundation':adapter.metadata,'selection':'first five fitting IDs; supplied checkpoints; no GT values used'}
        write_json(directory/'spec.json',spec);records={label:[] for label in labels};parameters={}
        for block,order in enumerate([list(range(len(labels))),list(range(len(labels)))[::-1]]):
            for index in order:
                label=labels[index];refiner=config=edge_cal=None
                if index:
                    refiner,config=load_refiner(paths[index-1])
                    parameters[label]=sum(p.numel() for p in refiner.parameters())
                    if needs_edges(config['candidate']):
                        if 'feature_cache' in config:edge_cal=config['feature_cache']['spec']['edge_calibration']
                        else:edge_cal=json.loads((ROOT/study['output']/'edge_features/metadata.json').read_text(encoding='utf-8'))['calibration']
                else:parameters[label]=0
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
                del refiner,config;gc.collect();torch.cuda.empty_cache()
        summary={}
        for label,blocks in records.items():
            values=np.array([x for block in blocks for x in block['seconds']])
            summary[label]={'calls':len(values),'mean_ms':float(values.mean()*1000),'median_ms':float(np.median(values)*1000),
                            'p95_ms':float(np.quantile(values,.95)*1000),'fps_inverse_mean':float(1/values.mean()),
                            'peak_allocated_mib':max(b['peak_allocated_mib'] for b in blocks),'refiner_parameters':parameters[label],
                            'block_mean_ms':[float(np.mean(b['seconds'])*1000) for b in blocks]}
        write_json(directory/'result.json',{'status':'complete','at':now(),'spec':spec,'summary':summary,
                   'limitations':'Two opposite orders reduce but do not remove laptop thermal/background-load variation; short local benchmark, not universal hardware speed.'})
        print('PASS isolated batch1 benchmark: '+str(directory),flush=True)


if __name__=='__main__':main()
