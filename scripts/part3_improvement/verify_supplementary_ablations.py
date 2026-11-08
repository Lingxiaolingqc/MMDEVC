"""Recompute supplementary predictions, weights, scores and check rendered timestamps."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import numpy as np
import av
from _common import ROOT,repo_path,checksum,write_json
from p3_depth.temporal.fusion import fuse_sequence
from part3_improvement.run_supplementary_ablations import (
    read_json,load_sources,specifications,reference_specs,describe)


def verify(run):
    meta=read_json(run/'metadata.json')
    if meta['status']!='complete' or meta['kind']!='supplementary_posthoc':raise ValueError('Incomplete supplement')
    experiment=meta['experiment']
    source=load_sources(*(repo_path(meta['sources'][name]) for name in ('cache','baseline','flow','reference')))
    if (meta['source_suite_sha256']!=checksum(source['reference']/'metadata.json') or
        meta['source_verification_sha256']!=checksum(source['reference']/'verification.json') or
        meta['specifications']!=specifications(experiment) or meta['parameters']!=source['sm']['config']['parameters'] or
        meta['normalization_limits']!=source['limits']):raise ValueError('Supplementary protocol/source changed')
    for path,digest in meta['artifacts_sha256'].items():
        if checksum(run/path)!=digest:raise ValueError('Supplementary artifact changed: '+path)
    expected=[]
    for spec in reference_specs(experiment)+specifications(experiment):
        reused=spec in reference_specs(experiment)
        folder=source['reference']/spec['variant'] if reused else run/spec['id']
        rm=read_json(folder/'metadata.json')
        pred=np.load(folder/'depths.npy',allow_pickle=False);valid=np.load(folder/'valid.npy',allow_pickle=False)
        weights=np.load(folder/'history_weight.npy',allow_pickle=False)
        result,w=fuse_sequence(source['base'],source['valid'],spec['variant'],source['sm']['config']['parameters'],source['fields'],
            representation='inverse_depth_like',history_source='recursive' if spec['variant']=='E0' else spec['history_source'],
            confidence_terms=spec['confidence_terms'])
        if not np.array_equal(pred,result,equal_nan=True) or not np.array_equal(weights,w):
            raise ValueError('Prediction or weight does not recompute: '+spec['id'])
        if not np.array_equal(valid,np.isfinite(result)):raise ValueError('Invalid mask differs: '+spec['id'])
        if not reused and (rm['ablation']!=spec or rm['parameters']!=meta['parameters'] or
                           rm['normalization_reference']['limits']!=source['limits']):
            raise ValueError('Per-run configuration differs')
        row,scores,pairs=describe(pred,valid,weights,spec,source,rm['fusion_seconds'],reused)
        if scores!=read_json(folder/'metrics.json'):raise ValueError('Scores do not recompute: '+spec['id'])
        with (folder/'per_pair.csv').open(encoding='utf-8',newline='') as handle:saved=list(csv.DictReader(handle))
        if saved!=[{k:'' if v is None else str(v) for k,v in r.items()} for r in pairs]:
            raise ValueError('Per-pair table differs: '+spec['id'])
        expected.append(row)
    with (run/'comparison.csv').open(encoding='utf-8',newline='') as handle:rows=list(csv.DictReader(handle))
    if rows!=[{k:'' if v is None else str(v) for k,v in r.items()} for r in expected]:raise ValueError('Comparison table differs')
    for spec in specifications(experiment):
        visual=run/'visualizations'/spec['id'];vm=read_json(visual/'metadata.json')
        refs=[source['reference']/spec['variant']] if experiment=='history' else [source['reference']/'E3',source['reference']/'E4']
        paths=refs+[run/spec['id']]
        if len(vm['inputs'])!=len(paths) or vm['frames']!=len(source['times']):raise ValueError('Visual comparison mismatch')
        for saved,path in zip(vm['inputs'],paths):
            if saved['metadata_sha256']!=checksum(path/'metadata.json') or [saved['low'],saved['high']]!=source['limits']:
                raise ValueError('Comparison changed sources or fixed color limits')
        for name,key in [('comparison.mp4','video_sha256'),('preview.png','preview_sha256')]:
            if checksum(visual/name)!=vm[key]:raise ValueError('Visualization changed')
        with av.open(str(visual/'comparison.mp4')) as container:times=[float(f.pts*f.time_base) for f in container.decode(video=0)]
        if len(times)!=len(source['times']) or not np.allclose(times,np.array(source['times'])-source['times'][0],rtol=0,atol=2e-6):
            raise ValueError('Visualization timestamps differ')
    return {'status':'verified','experiment':experiment,'rows':len(rows),'frames':len(source['times']),
            'supplement_metadata_sha256':checksum(run/'metadata.json'),
            'statement':'All reused/new arrays, weights, pair metrics and tables recomputed. Four comparisons have unchanged sources, shared fixed color limits and original timestamps. Not visual acceptance or depth-accuracy evidence.',
            'verifier_sha256':checksum(ROOT/'scripts/part3_improvement/verify_supplementary_ablations.py')}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True)
    p.add_argument('--verification-output',help='Optional new receipt path; never replaces an existing receipt')
    a=p.parse_args();run=repo_path(a.run)
    receipt=repo_path(a.verification_output) if a.verification_output else run/'verification.json'
    if receipt.exists():raise FileExistsError('Receipt already exists; use a new --verification-output')
    result=verify(run);write_json(receipt,result)
    print(f"PASS {result['experiment']}: {result['rows']} rows, saved arrays/weights/scores and video timestamps recomputed")


if __name__=='__main__':main()
