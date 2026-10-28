"""Run DA V2 or VDA on the same immutable timestamped RGB cache."""

import argparse
import json
import subprocess
from time import perf_counter
import numpy as np

from run_stereo_baseline import ROOT, repo_path, read_yaml, write_json, checksum
from p3_depth.models.runtime import ImageModel
from p3_depth.models.video import VideoModel


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['dav2_relative','dav2_metric','vda_metric'],required=True)
    p.add_argument('--cache',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--paths-config',default='configs/paths.local.yaml')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--precision',choices=['fp32','fp16'],default='fp16')
    p.add_argument('--input-size',type=int,default=364)
    p.add_argument('--threads',type=int,default=8)
    args=p.parse_args()
    import torch
    cache, output=repo_path(args.cache),repo_path(args.output_dir)
    if output.exists():
        raise FileExistsError('Use a new output directory')
    meta=json.loads((cache/'metadata.json').read_text(encoding='utf-8'))
    if checksum(cache/'frames.npy') != meta['frames_sha256'] or checksum(cache/'frames.csv') != meta['index_sha256']:
        raise ValueError('Video cache hash mismatch')
    frames=np.load(cache/'frames.npy',mmap_mode='r',allow_pickle=False)
    settings=read_yaml(repo_path(args.paths_config))
    if args.model=='vda_metric':
        model=VideoModel(ROOT,settings,args.device,args.precision,args.input_size,args.threads)
        warm=model(np.array(frames[:1]),meta['requested_fps'])
    else:
        model=ImageModel(args.model,ROOT,settings,args.device,args.input_size,args.precision,args.threads)
        warm=model(np.array(frames[0]))
    del warm
    if args.device=='cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start=perf_counter()
    if args.model=='vda_metric':
        depths=model(np.asarray(frames),meta['requested_fps'])
    else:
        depths=np.stack([model(np.array(frame)) for frame in frames])
    if args.device=='cuda':
        torch.cuda.synchronize()
    elapsed=perf_counter()-start
    valid=np.isfinite(depths)
    if args.model!='dav2_relative':
        valid &= depths>0
    output.mkdir(parents=True)
    np.save(output/'depths.npy',depths)
    np.save(output/'valid.npy',valid)
    info={'status':'complete','model':model.metadata,'cache_metadata':meta,'cache_path':str(cache),
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'working_tree_status':subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),
        'implementation_sha256':{name:checksum(ROOT/name) for name in ['scripts/run_video_models.py','src/p3_depth/models/video.py','src/p3_depth/models/runtime.py']},
        'frames':len(depths),'prediction_type':'inverse_depth_like' if args.model=='dav2_relative' else 'metric_depth',
        'unit':'arbitrary' if args.model=='dav2_relative' else 'm','alignment':'none','gt_used':False,
        'depths_sha256':checksum(output/'depths.npy'),'valid_sha256':checksum(output/'valid.npy'),
        'inference_seconds':elapsed,'mean_seconds_per_frame':elapsed/len(depths),'fps':len(depths)/elapsed,
        'timing_scope':'synchronized cache RGB to raw depth, official preprocessing and temporal stitching included; excludes load, decode and disk writes',
        'warmup':'one original frame; VDA internally pads its official temporal window',
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if args.device=='cuda' else None,
        'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved() if args.device=='cuda' else None}
    write_json(output/'metadata.json',info)
    print(f'PASS {args.model}: {len(depths)} original-sized outputs; {elapsed:.3f}s; scope={meta["scope"]}',flush=True)


if __name__=='__main__':
    main()
