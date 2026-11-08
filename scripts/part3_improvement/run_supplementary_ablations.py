"""Post-hoc history/confidence controls from verified saved outputs, without inference."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
from time import perf_counter
import numpy as np
from _common import ROOT,repo_path,checksum,write_csv,write_json
from _paths import clip_output
from _media import checked_comparison
from p3_depth.temporal.fusion import fuse_sequence,CONFIDENCE_TERMS
from p3_depth.temporal.selection import suite_artifacts
from part3_improvement.run_fusion_suite import load_run,suite_scores,score_columns


def read_json(path):return json.loads(path.read_text(encoding='utf-8'))


def specifications(experiment):
    if experiment=='history':
        return [{'id':v+'_pairwise','variant':v,'history_source':'pairwise','confidence_terms':list(CONFIDENCE_TERMS) if v=='E4' else []}
                for v in ('E1','E2','E3','E4')]
    if experiment=='confidence':
        return [{'id':'E4_no_'+term,'variant':'E4','history_source':'recursive',
                 'confidence_terms':[t for t in CONFIDENCE_TERMS if t!=term]} for term in CONFIDENCE_TERMS]
    raise ValueError('Unknown supplementary experiment')


def load_sources(cache,baseline,flow,reference):
    cm=read_json(cache/'metadata.json');sm=read_json(reference/'metadata.json')
    if sm['status']!='complete' or sm['prediction_type']!='inverse_depth_like':
        raise ValueError('Require a completed relative-depth E0-E5 reference suite')
    for path,key in [(cache,'cache_metadata_sha256'),(baseline,'base_metadata_sha256'),(flow,'flow_metadata_sha256')]:
        if checksum(path/'metadata.json')!=sm[key]:raise ValueError('Reference suite uses different source metadata')
    receipt=read_json(reference/'verification.json')
    if receipt['status']!='verified' or receipt['suite_metadata_sha256']!=checksum(reference/'metadata.json'):
        raise ValueError('Reference suite must have a matching verification receipt')
    if set(receipt['artifacts_sha256'])!=set(suite_artifacts()):
        raise ValueError('Incomplete reference verification receipt')
    # Existing E0-E5 predictions/metrics/weights stay immutable, including unused VDA scores.
    for name,digest in receipt['artifacts_sha256'].items():
        if checksum(reference/name)!=digest:raise ValueError('Verified reference artifact changed: '+name)
    for name,key in [('frames.npy','frames_sha256'),('frames.csv','index_sha256')]:
        if checksum(cache/name)!=cm[key]:raise ValueError('RGB/time cache changed')
    bm,base,valid=load_run(baseline,cm)
    if bm['model']['model']!='dav2_relative' or bm['prediction_type']!='inverse_depth_like':
        raise ValueError('Supplementary controls require the original DA relative output')
    fm=read_json(flow/'metadata.json')
    if fm['status']!='complete' or fm['cache_metadata']!=cm:raise ValueError('Flow/cache identity mismatch')
    fields={}
    for name,digest in fm['artifacts_sha256'].items():
        if checksum(flow/(name+'.npy'))!=digest:raise ValueError('Saved flow changed')
        fields[name]=np.load(flow/(name+'.npy'),mmap_mode='r',allow_pickle=False)
    with (cache/'frames.csv').open(encoding='utf-8',newline='') as handle:
        times=[float(r['timestamp_s']) for r in csv.DictReader(handle)]
    limits=list(map(float,np.quantile(base[valid],[.01,.99])))
    if limits!=sm['baseline_normalization_limits'] or not limits[1]>limits[0]:
        raise ValueError('Invalid or changed reference normalization span')
    shape=np.load(cache/'frames.npy',mmap_mode='r',allow_pickle=False).shape[:3]
    if shape!=base.shape or len(times)!=len(base) or not np.all(np.diff(times)>0):
        raise ValueError('Prediction/cache/timestamp dimensions differ')
    if sm['split']=='test' and sm['config']['status']!='frozen':
        raise ValueError('Test reference must be the existing frozen result')
    return {'cache':cache,'baseline':baseline,'flow':flow,'reference':reference,'cm':cm,'sm':sm,'bm':bm,'fm':fm,
            'base':base,'valid':valid,'fields':fields,'times':times,'limits':limits}


def describe(pred,valid,weights,spec,source,elapsed,reused):
    scores,pairs=suite_scores(pred,valid,source['fields'],source['times'],True,source['cm']['scene_type']=='static',source['limits'])
    reference=source['fields']['common_mask']
    selected=reference & valid[1:]
    denominator=int(reference.sum());finite=selected & np.isfinite(weights[1:])
    span=source['limits'][1]-source['limits'][0]
    raw_valid=valid & source['valid']
    change=np.abs(pred[raw_valid].astype(np.float64)-source['base'][raw_valid])/span
    flow_seconds=source['fm']['elapsed_seconds'] if spec['variant'] in ('E2','E3','E4') else 0.
    total=source['bm']['inference_seconds']+flow_seconds+elapsed
    row={'id':spec['id'],'variant':spec['variant'],'history_source':spec['history_source'],
         'confidence_terms':'+'.join(spec['confidence_terms']),'reused_original':reused,
         **{column:scores[key] for column,key in score_columns(True).items()},
         'pair_coverage':scores['pair_coverage'],'reference_pixels':scores['reference_pixels'],
         'mean_history_weight_on_reference':float(weights[1:][finite].astype(np.float64).mean()) if finite.any() else None,
         'history_used_fraction_on_reference':int(((weights[1:]>0)&selected).sum())/denominator if denominator else None,
         'mean_departure_from_DA_normalized':float(change.mean()) if change.size else None,
         'fusion_seconds':elapsed,'flow_seconds':flow_seconds,'estimated_pipeline_fps':len(pred)/total}
    return row,scores,pairs


def reference_specs(experiment):
    variants=('E0','E1','E2','E3','E4') if experiment=='history' else ('E0','E3','E4')
    return [{'id':v+'_recursive' if v!='E0' else 'E0','variant':v,
             'history_source':'recursive' if v!='E0' else 'none',
             'confidence_terms':list(CONFIDENCE_TERMS) if v=='E4' else []} for v in variants]


def render_supplement(source,output,experiment):
    for spec in specifications(experiment):
        references=[source['reference']/spec['variant']] if experiment=='history' else [source['reference']/'E3',source['reference']/'E4']
        checked_comparison(source['cache'],references+[output/spec['id']],output/'visualizations'/spec['id'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','baseline','flow','reference-suite'):p.add_argument('--'+name,required=True)
    p.add_argument('--experiment',choices=('history','confidence'),required=True)
    p.add_argument('--output-dir')
    p.add_argument('--visualize-only',action='store_true')
    a=p.parse_args()
    source=load_sources(*(repo_path(getattr(a,name)) for name in ('cache','baseline','flow','reference_suite')))
    out=clip_output('fusion_supplement',source['cm'],a.output_dir,experiment=a.experiment)
    if a.visualize_only:
        meta=read_json(out/'metadata.json')
        if meta['status']!='complete' or meta['experiment']!=a.experiment or meta['source_suite_sha256']!=checksum(source['reference']/'metadata.json'):
            raise ValueError('Use the same completed supplementary experiment')
        for path,digest in meta['artifacts_sha256'].items():
            if checksum(out/path)!=digest:raise ValueError('Supplementary artifact changed')
        render_supplement(source,out,a.experiment);return
    if out.exists():raise FileExistsError('Use a new supplementary directory; do not overwrite the original suite')
    out.mkdir(parents=True)
    rows=[]
    for spec in reference_specs(a.experiment):
        folder=source['reference']/spec['variant'];meta=read_json(folder/'metadata.json')
        pred=np.load(folder/'depths.npy',mmap_mode='r',allow_pickle=False)
        valid=np.load(folder/'valid.npy',mmap_mode='r',allow_pickle=False)
        weights=np.load(folder/'history_weight.npy',mmap_mode='r',allow_pickle=False)
        row,scores,_=describe(pred,valid,weights,spec,source,meta['fusion_seconds'],True)
        if scores!=read_json(folder/'metrics.json'):raise ValueError('Reference scores do not recompute')
        rows.append(row)
    for spec in specifications(a.experiment):
        start=perf_counter()
        pred,weights=fuse_sequence(source['base'],source['valid'],spec['variant'],source['sm']['config']['parameters'],
            source['fields'],representation='inverse_depth_like',history_source=spec['history_source'],confidence_terms=spec['confidence_terms'])
        elapsed=perf_counter()-start;valid=np.isfinite(pred)
        folder=out/spec['id'];folder.mkdir()
        for name,array in [('depths',pred),('valid',valid),('history_weight',weights)]:np.save(folder/(name+'.npy'),array)
        row,scores,pairs=describe(pred,valid,weights,spec,source,elapsed,False);rows.append(row)
        write_json(folder/'metrics.json',scores);write_csv(folder/'per_pair.csv',pairs)
        write_json(folder/'metadata.json',{'status':'complete','model':{'model':spec['id']},'cache_metadata':source['cm'],
            'prediction_type':'inverse_depth_like','unit':'arbitrary','alignment':'none','gt_used':False,
            'normalization_reference':{'limits':source['limits'],'source_metadata_sha256':checksum(source['baseline']/'metadata.json')},
            **{name+'_sha256':checksum(folder/(name+'.npy')) for name in ('depths','valid','history_weight')},
            'parameters':source['sm']['config']['parameters'],'ablation':spec,'fusion_seconds':elapsed,
            'config_status':'supplementary_posthoc','source_suite_sha256':checksum(source['reference']/'metadata.json')})
        print('PASS saved '+spec['id'],flush=True)
    if len({r['reference_pixels'] for r in rows})!=1:raise ValueError('Reference denominators differ')
    write_csv(out/'comparison.csv',rows)
    write_json(out/'metadata.json',{'status':'complete','experiment':a.experiment,'kind':'supplementary_posthoc',
        'scope':source['cm']['scope'],'clip_id':source['cm']['clip_id'],'split':source['cm']['split'],
        'sources':{name:source[name].relative_to(ROOT).as_posix() for name in ('cache','baseline','flow','reference')},
        'source_suite_sha256':checksum(source['reference']/'metadata.json'),
        'source_verification_sha256':checksum(source['reference']/'verification.json'),
        'specifications':specifications(a.experiment),'parameters':source['sm']['config']['parameters'],
        'normalization_limits':source['limits'],'raw_sources_modified':False,
        'interpretation':'Post-hoc controlled analysis, not a fresh blind test or a replacement of the frozen main result. Temporal change, departure and history weights are diagnostics, not depth accuracy or quality scores.',
        'confidence_ablation':'Only soft exponent terms are disabled; hard FB and RGB gates stay enabled. History comparison keeps the weight rule and parameters, not necessarily the resulting per-pixel weights.',
        'efficiency':'Original model and flow times plus each fusion time; additive estimate excluding IO. Reused references and new variants were not timed in a shared benchmark session.',
        'artifacts_sha256':{path.relative_to(out).as_posix():checksum(path) for path in sorted(out.rglob('*')) if path.is_file()},
        'implementation_sha256':{name:checksum(ROOT/name) for name in ['scripts/part3_improvement/run_supplementary_ablations.py','src/p3_depth/temporal/fusion.py','src/p3_depth/evaluation/temporal.py']}})
    render_supplement(source,out,a.experiment)
    print(f'PASS {a.experiment}: {len(rows)} matched rows; four new configurations; {out}')


if __name__=='__main__':main()
