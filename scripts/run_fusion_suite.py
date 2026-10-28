"""Run causal E0-E4, compare independent E5, and score all on the same RGB-only mask."""
import argparse
import csv
import json
from time import perf_counter
import numpy as np

from run_stereo_baseline import ROOT,repo_path,read_yaml,write_json,write_csv,checksum
from p3_depth.temporal.fusion import fuse_sequence
from p3_depth.evaluation.temporal import evaluate_video
from p3_depth.temporal.selection import validate_test_freeze


def load_run(path,cache_meta):
    meta=json.loads((path/'metadata.json').read_text(encoding='utf-8'))
    if meta['status']!='complete' or meta['prediction_type']!='metric_depth' or meta['unit']!='m' or meta['alignment']!='none':
        raise ValueError('Fusion comparisons require completed unaligned metric Z predictions in m')
    for key in ('frames_sha256','index_sha256'):
        if meta['cache_metadata'][key]!=cache_meta[key]:
            raise ValueError('All methods must use the same RGB frames and timestamps')
    for name,key in [('depths.npy','depths_sha256'),('valid.npy','valid_sha256')]:
        if checksum(path/name)!=meta[key]:
            raise ValueError('Prediction artifact hash mismatch')
    return meta,np.load(path/'depths.npy',mmap_mode='r',allow_pickle=False),np.load(path/'valid.npy',mmap_mode='r',allow_pickle=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',required=True)
    p.add_argument('--baseline',required=True)
    p.add_argument('--reference-vda',required=True)
    p.add_argument('--flow',required=True)
    p.add_argument('--config',default='configs/fusion_provisional.yaml')
    p.add_argument('--output-dir',required=True)
    a=p.parse_args()
    cache,output,flow_dir=repo_path(a.cache),repo_path(a.output_dir),repo_path(a.flow)
    if output.exists():
        raise FileExistsError('Use a new suite directory')
    cm=json.loads((cache/'metadata.json').read_text(encoding='utf-8'))
    for name,key in [('frames.npy','frames_sha256'),('frames.csv','index_sha256')]:
        if checksum(cache/name)!=cm[key]:
            raise ValueError('Video cache hash mismatch')
    config=read_yaml(repo_path(a.config))
    if cm['split']=='test':
        validate_test_freeze(ROOT,config,cm)
    bm,base,valid=load_run(repo_path(a.baseline),cm)
    vm,vda,vda_valid=load_run(repo_path(a.reference_vda),cm)
    if bm['model']['model']!='dav2_metric' or vm['model']['model']!='vda_metric':
        raise ValueError('E0-E4 use DA metric; E5 is independent raw metric VDA')
    for key in ('device','device_name'):
        if bm['model'][key]!=vm['model'][key]:
            raise ValueError('Compare models on the same device')
    if bm['model']['requested_precision']!=vm['model']['precision']:
        raise ValueError('Compare models with the same inference precision')
    if {tuple(s[-2:]) for s in bm['model']['encoder_shapes_nchw']}!={tuple(s[-2:]) for s in vm['model']['encoder_shapes_nchw']}:
        raise ValueError('DA and VDA actual spatial encoder dimensions must match')
    fm=json.loads((flow_dir/'metadata.json').read_text(encoding='utf-8'))
    if fm['status']!='complete' or any(fm['cache_metadata'][k]!=cm[k] for k in ('frames_sha256','index_sha256')):
        raise ValueError('Flow cache must use the same frames and times')
    fields={}
    for key,expected in fm['artifacts_sha256'].items():
        path=flow_dir/(key+'.npy')
        if checksum(path)!=expected:
            raise ValueError('Flow artifact hash mismatch')
        fields[key]=np.load(path,mmap_mode='r',allow_pickle=False)
    with (cache/'frames.csv').open(encoding='utf-8',newline='') as f:
        times=[float(r['timestamp_s']) for r in csv.DictReader(f)]
    if base.shape!=vda.shape or base.shape[0]!=len(times):
        raise ValueError('Mismatched depth sequences')
    output.mkdir(parents=True)
    table=[]
    for variant in ('E0','E1','E2','E3','E4','E5'):
        folder=output/variant;folder.mkdir()
        if variant=='E5':
            prediction,pred_valid=vda,vda_valid
            elapsed=0.;source=vm
        else:
            start=perf_counter()
            prediction,weight=fuse_sequence(base,valid,variant,config['parameters'],fields)
            elapsed=perf_counter()-start
            pred_valid=np.isfinite(prediction)&(prediction>0);source=bm
            np.save(folder/'depths.npy',prediction);np.save(folder/'valid.npy',pred_valid)
            np.save(folder/'history_weight.npy',weight)
            write_json(folder/'metadata.json',{'status':'complete','model':{'model':variant},'cache_metadata':cm,
                'prediction_type':'metric_depth','unit':'m','alignment':'none','gt_used':False,
                'depths_sha256':checksum(folder/'depths.npy'),'valid_sha256':checksum(folder/'valid.npy'),
                'history_weight_sha256':checksum(folder/'history_weight.npy'),
                'parameters':config['parameters'],'config_status':config['status'],'base_metadata_sha256':checksum(repo_path(a.baseline)/'metadata.json'),
                'flow_metadata_sha256':checksum(flow_dir/'metadata.json'),'fusion_seconds':elapsed})
        scores,rows=evaluate_video(prediction,pred_valid,fields['backward'],fields['common_mask'],times,static_scene=cm['scene_type']=='static')
        write_json(folder/'metrics.json',scores);write_csv(folder/'per_pair.csv',rows)
        flow_seconds=fm['elapsed_seconds'] if variant in ('E2','E3','E4') else 0.
        total=source['inference_seconds']+flow_seconds+elapsed
        table.append({'variant':variant,'temporal_l1_m':scores['temporal_l1_m_on_valid'],
            'capped_l1_m_full_reference':scores['capped_l1_m_full_reference'],
            'agreement_5pct_full_reference':scores['agreement_5pct_full_reference'],
            'pair_coverage':scores['pair_coverage'],'reference_pixels':scores['reference_pixels'],
            'median_abs_log_change':scores['median_abs_log_change'],'static_pixel_std_m':scores['static_pixel_std_m'],
            'fusion_seconds':elapsed,'flow_seconds':flow_seconds,'estimated_pipeline_fps':len(base)/total})
    if len({r['reference_pixels'] for r in table})!=1:
        raise ValueError('Methods did not use identical reference denominators')
    write_csv(output/'comparison.csv',table)
    write_json(output/'metadata.json',{'status':'complete','scope':cm['scope'],'split':cm['split'],'clip_id':cm['clip_id'],
        'cache_metadata_sha256':checksum(cache/'metadata.json'),'base_metadata_sha256':checksum(repo_path(a.baseline)/'metadata.json'),
        'vda_metadata_sha256':checksum(repo_path(a.reference_vda)/'metadata.json'),'flow_metadata_sha256':checksum(flow_dir/'metadata.json'),
        'config':config,'config_sha256':checksum(repo_path(a.config)),
        'comparison_sha256':checksum(output/'comparison.csv'),'E5_predictions':a.reference_vda,
        'efficiency':'estimated additive base inference + CPU flow + fusion; excludes decode/IO; not measured end-to-end FPS',
        'implementation_sha256':{name:checksum(ROOT/name) for name in ['scripts/run_fusion_suite.py','src/p3_depth/temporal/fusion.py','src/p3_depth/temporal/flow.py','src/p3_depth/evaluation/temporal.py']}})
    print(f'PASS E0-E5: same {table[0]["reference_pixels"]} reference pixels; {cm["scope"]}; {config["status"]}')


if __name__=='__main__':
    main()
