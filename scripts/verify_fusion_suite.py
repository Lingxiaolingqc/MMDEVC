"""Recompute saved flow, E0-E4 fusion and all E0-E5 scores without model inference."""
import argparse
import csv
import json
import cv2
import numpy as np

from run_stereo_baseline import ROOT,repo_path,checksum,write_json
from run_fusion_suite import load_run
from p3_depth.temporal.selection import suite_artifacts,relative
from p3_depth.temporal.flow import estimate_pair
from p3_depth.temporal.fusion import fuse_sequence
from p3_depth.evaluation.temporal import evaluate_video


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('suite','cache','baseline','reference-vda','flow'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args()
    suite,cache,flow_dir=repo_path(a.suite),repo_path(a.cache),repo_path(a.flow)
    sm=json.loads((suite/'metadata.json').read_text(encoding='utf-8'))
    sources=[(cache,'cache_metadata_sha256'),(repo_path(a.baseline),'base_metadata_sha256'),
             (repo_path(a.reference_vda),'vda_metadata_sha256'),(flow_dir,'flow_metadata_sha256')]
    if sm['status']!='complete':
        raise ValueError('Suite is not complete')
    for path,key in sources:
        if checksum(path/'metadata.json')!=sm[key]:
            raise ValueError('Suite source metadata changed')
    cm=json.loads((cache/'metadata.json').read_text(encoding='utf-8'))
    for name,key in [('frames.npy','frames_sha256'),('frames.csv','index_sha256')]:
        if checksum(cache/name)!=cm[key]:
            raise ValueError('RGB/time cache changed')
    frames=np.load(cache/'frames.npy',mmap_mode='r',allow_pickle=False)
    fm=json.loads((flow_dir/'metadata.json').read_text(encoding='utf-8'))
    fields={}
    for key,h in fm['artifacts_sha256'].items():
        path=flow_dir/(key+'.npy')
        if checksum(path)!=h:
            raise ValueError('Flow artifact changed')
        fields[key]=np.load(path,mmap_mode='r',allow_pickle=False)
    cv2.setNumThreads(fm['config']['threads'])
    for t in range(1,len(frames)):
        forward,backward,geometry=estimate_pair(frames[t-1],frames[t],fm['config'])
        for key,array in {'forward':forward,'backward':backward,**{k:geometry[k] for k in ('fb_residual','photo_residual','common_mask')}}.items():
            if not np.allclose(array,fields[key][t-1],rtol=1e-5,atol=1e-5,equal_nan=True):
                raise ValueError(f'Flow/mask recomputation differs: {t} {key}')
    _,base,valid=load_run(repo_path(a.baseline),cm)
    _,vda,vda_valid=load_run(repo_path(a.reference_vda),cm)
    with (cache/'frames.csv').open(encoding='utf-8',newline='') as f:
        times=[float(r['timestamp_s']) for r in csv.DictReader(f)]
    if checksum(suite/'comparison.csv')!=sm['comparison_sha256']:
        raise ValueError('Comparison table changed')
    with (suite/'comparison.csv').open(encoding='utf-8',newline='') as f:
        table=list(csv.DictReader(f))
    if [r['variant'] for r in table]!=['E0','E1','E2','E3','E4','E5']:
        raise ValueError('Incomplete ablation table')
    for variant in ('E0','E1','E2','E3','E4','E5'):
        folder=suite/variant
        if variant=='E5':
            pred,pred_valid=vda,vda_valid
        else:
            expected,weights=fuse_sequence(base,valid,variant,sm['config']['parameters'],fields)
            meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
            for name,key in [('depths.npy','depths_sha256'),('valid.npy','valid_sha256'),('history_weight.npy','history_weight_sha256')]:
                if checksum(folder/name)!=meta[key]:
                    raise ValueError(f'{variant} artifact changed')
            pred=np.load(folder/'depths.npy',allow_pickle=False)
            pred_valid=np.load(folder/'valid.npy',allow_pickle=False)
            saved_weights=np.load(folder/'history_weight.npy',allow_pickle=False)
            if not np.array_equal(expected,pred,equal_nan=True) or not np.array_equal(weights,saved_weights):
                raise ValueError(f'{variant} fusion recomputation differs')
            if not np.array_equal(pred_valid,np.isfinite(pred)&(pred>0)):
                raise ValueError('Invalid prediction mask')
        scores,_=evaluate_video(pred,pred_valid,fields['backward'],fields['common_mask'],times,static_scene=cm['scene_type']=='static')
        if scores!=json.loads((folder/'metrics.json').read_text(encoding='utf-8')):
            raise ValueError(f'{variant} metric recomputation differs')
        row=next(r for r in table if r['variant']==variant)
        for column,key in [('temporal_l1_m','temporal_l1_m_on_valid'),('capped_l1_m_full_reference','capped_l1_m_full_reference'),('reference_pixels','reference_pixels'),('pair_coverage','pair_coverage')]:
            actual=float(row[column]) if row[column] else None
            if actual!=scores[key]:
                raise ValueError('Exported table differs from verified scores')
    write_json(suite/'verification.json',{'schema_version':1,'status':'verified',
        'suite_metadata_sha256':checksum(suite/'metadata.json'),
        'artifacts_sha256':{name:checksum(suite/name) for name in suite_artifacts()},
        'input_metadata_sha256':{relative(ROOT,path/'metadata.json'):checksum(path/'metadata.json') for path,_ in sources},
        'verifier_sha256':checksum(ROOT/'scripts/verify_fusion_suite.py'),
        'scope':sm['scope'],'statement':'Recomputed flow/masks, E0-E4 fusion/weights and E0-E5 metrics; not member acceptance'})
    print(f'PASS: {len(frames)-1} flow/mask pairs, E0-E4 raw fusion/weights, and E0-E5 metrics recomputed; scope={sm["scope"]}')


if __name__=='__main__':
    main()
