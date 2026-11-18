"""Diagnose frozen confidence/flip disagreement against depth error on development only."""
from pathlib import Path
import json
import sys
import csv
import h5py
import numpy as np
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from p3_depth.data.nyuv2 import NYUv2Labeled
from p3_depth.residual.records import sha,write_json,snapshot


def main():
    study=json.loads((ROOT/'configs/residual_study.json').read_text(encoding='utf-8'))
    directory=ROOT/study['output']
    output=directory/'input_diagnostics'
    output.mkdir(exist_ok=True)
    if (output/'review.json').exists():
        raise FileExistsError('Existing diagnostic results are preserved')
    snapshot(ROOT,output)
    split=json.loads((ROOT/study['split']).read_text(encoding='utf-8'))
    meta=json.loads((directory/'cache/development/metadata.json').read_text(encoding='utf-8'))
    source_path=directory/'cache/development/predictions.h5'
    if meta['status']!='complete' or sha(source_path)!=meta['cache_sha256']:
        raise ValueError('Unverified development cache')
    # Fixed 8-pixel lattice gives a bounded, reproducible correlation diagnostic.
    # Quantiles are calibrated on fitting only; no thresholds are selected by correlation.
    data={k:[] for k in ('fitting','validation')}
    with NYUv2Labeled(ROOT/study['data']) as nyu,h5py.File(source_path,'r') as cache:
        for group,ids in [('fitting',split['fit_ids']),('validation',split['validation_ids'])]:
            for n,index in enumerate(ids,1):
                sample=nyu.load(index+1)
                key=cache[str(index)]
                region=np.zeros((480,640),bool)
                region[45:471:8,41:601:8]=True
                region &= sample.valid_gt
                raw=key['original'][:][region].astype(np.float64)
                flip=key['flip'][:][region].astype(np.float64)
                confidence=key['confidence'][:][region].astype(np.float64)
                gt=sample.depth_m[region].astype(np.float64)
                values=np.stack([np.log(np.maximum(confidence,1e-6)),np.abs(np.log(flip)-np.log(raw)),
                                 np.abs(np.clip(raw,.001,10)-gt)/gt])
                data[group].append(values)
                if n%100==0 or n==len(ids):
                    print(f'DIAGNOSE {group} {n}/{len(ids)}',flush=True)
    arrays={key:np.concatenate(values,axis=1) for key,values in data.items()}
    result={'test_used':False,'cache_sha256':meta['cache_sha256'],
            'sampling':'fixed lattice every8 pixels anchored at crop origin (45,41), valid GT only; descriptive pixel-pooled diagnostic',
            'warning':'confidence is the official output field, not assumed calibrated correctness probability; disagreement is not GT error',
            'features':{}}
    rows=[]
    for axis,name in enumerate(('log_official_confidence','absolute_log_flip_disagreement')):
        thresholds=np.quantile(arrays['fitting'][axis],[.25,.5,.75])
        item={'quartile_thresholds_fitting_only':thresholds.tolist(),'split_statistics':{}}
        for group,values in arrays.items():
            correlation=spearmanr(values[axis],values[2]).statistic
            bins=np.searchsorted(thresholds,values[axis],side='right')
            groups=[]
            for i in range(4):
                selected=bins==i
                row={'feature':name,'split':group,'quartile':i+1,'pixels':int(selected.sum()),
                     'mean_abs_rel':float(values[2,selected].mean()) if selected.any() else None,
                     'feature_mean':float(values[axis,selected].mean()) if selected.any() else None}
                rows.append(row);groups.append(row)
            item['split_statistics'][group]={'pixels':values.shape[1],'spearman_with_abs_rel':float(correlation),'quartiles':groups}
        result['features'][name]=item
    write_json(output/'review.json',result)
    with (output/'quartiles.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Input diagnostic','',result['sampling'],'',result['warning'],'',
           'Quartile boundaries use fitting data only. Correlation is descriptive; neighboring pixels are dependent, so no pixel-independent significance claim is made. These values do not prove a feature improves a trained refiner.','']
    for name,item in result['features'].items():
        vals=item['split_statistics']['validation']
        lines.append(f'- {name}: validation Spearman vs absolute relative error = {vals["spearman_with_abs_rel"]:.4f}; quartile mean errors = '+', '.join(f'{x["mean_abs_rel"]:.4f}' for x in vals['quartiles']))
    (output/'review.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('PASS input diagnostics; no test images, no learned-model conclusion',flush=True)


if __name__=='__main__':
    main()
