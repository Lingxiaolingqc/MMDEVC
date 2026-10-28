"""Cache RGB-only forward/backward flow and one fixed evaluation mask per pair."""
import argparse
import json
from time import perf_counter
import cv2
import numpy as np

from run_stereo_baseline import ROOT,repo_path,read_yaml,write_json,checksum
from p3_depth.temporal.flow import estimate_pair


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',required=True)
    p.add_argument('--config',default='configs/flow.yaml')
    p.add_argument('--output-dir',required=True)
    a=p.parse_args()
    cache,output=repo_path(a.cache),repo_path(a.output_dir)
    if output.exists():
        raise FileExistsError('Use a new flow directory')
    meta=json.loads((cache/'metadata.json').read_text(encoding='utf-8'))
    for name,key in [('frames.npy','frames_sha256'),('frames.csv','index_sha256')]:
        if checksum(cache/name)!=meta[key]:
            raise ValueError('RGB/time cache hash mismatch')
    frames=np.load(cache/'frames.npy',mmap_mode='r',allow_pickle=False)
    settings=read_yaml(repo_path(a.config))
    if settings['algorithm']!='opencv_farneback':
        raise ValueError('Unsupported flow algorithm')
    cv2.setNumThreads(settings['threads'])
    fields={name:[] for name in ('forward','backward','fb_residual','photo_residual','common_mask')}
    start=perf_counter()
    for t in range(1,len(frames)):
        f,b,info=estimate_pair(frames[t-1],frames[t],settings)
        fields['forward'].append(f);fields['backward'].append(b)
        for key in ('fb_residual','photo_residual','common_mask'):
            fields[key].append(info[key])
    elapsed=perf_counter()-start
    output.mkdir(parents=True)
    hashes={}
    for key,value in fields.items():
        path=output/(key+'.npy');np.save(path,np.stack(value));hashes[key]=checksum(path)
    write_json(output/'metadata.json',{'status':'complete','cache_metadata':meta,'config':settings,
        'config_sha256':checksum(repo_path(a.config)),'artifacts_sha256':hashes,
        'pairs':len(frames)-1,'elapsed_seconds':elapsed,'seconds_per_pair':elapsed/(len(frames)-1),
        'opencv':cv2.__version__,'unit':'pixels at cached RGB resolution',
        'coordinates':'forward[t-1]: previous -> current on previous grid; backward[t-1]: current -> previous on current grid',
        'evaluation_region':'current-frame pixels; RGB/flow only; independent of all depth predictions and fusion weights',
        'implementation_sha256':{name:checksum(ROOT/name) for name in ['scripts/prepare_flow.py','src/p3_depth/temporal/flow.py']}})
    print(f'PASS: {len(frames)-1} RGB-only flow pairs; {elapsed:.3f}s; {output}')


if __name__=='__main__':
    main()
