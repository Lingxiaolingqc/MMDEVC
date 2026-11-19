"""Matched-factor analysis and validation-only Residual selection."""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from analyze_residual_study import read_csv,mean_seed_scores,scene_bootstrap,benefit_distribution
from p3_depth.adaptation_study.common import load
from p3_depth.residual.records import sha,write_json,now

KEYS=['abs_rel','rmse_m','delta1','boundary_f1','boundary_band_abs_rel']


def collect(study):
    records=[]
    for seed in study['confirmation_seeds']:
        d=ROOT/study['parent_output']/'runs'/f'frozen_features_seed{seed}'
        records.append({'name':'S0','seed':seed,'path':d,
                        'result':json.loads((d/'result.json').read_text()),
                        'scores':read_csv(d/'validation/per_image.csv')})
    for path in sorted((ROOT/study['output']/'followups').glob('*/result.json')):
        result=json.loads(path.read_text())
        cfg=json.loads((path.parent/'config.json').read_text())
        if result['smoke'] or result['phase'] not in ('factorial','control','zero_middle'):
            continue
        records.append({'name':cfg['candidate']['name'],'seed':cfg['seed'],'path':path.parent,
                        'result':result,'scores':read_csv(path.parent/'validation/per_image.csv')})
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',action='store_true')
    args=parser.parse_args()
    study,_=load(ROOT);out=ROOT/study['output'];groups={}
    for row in collect(study):groups.setdefault(row['name'],[]).append(row)
    summary=[]
    for name,rows in groups.items():
        assert len({r['seed'] for r in rows})==len(rows)
        values={'method':name,'seeds':sorted(r['seed'] for r in rows),'n_seeds':len(rows),
                'parameters':rows[0]['result']['parameters']}
        for key in KEYS:
            scores=[r['result']['validation'][key] for r in rows]
            values[key]=float(np.mean(scores))
            values[key+'_sd']=float(np.std(scores,ddof=1)) if len(scores)>1 else None
        values['best_epochs']=[r['result'].get('best_unit',r['result'].get('best_epoch')) for r in rows]
        summary.append(values)
    definitions=[('fusion','F2','P2'),('fusion','F3','P3'),('gate','P2','G2'),('gate','P3','G3'),
                 ('stage_count','F2','F3'),('stage_count','P2','P3'),('stage_count','G2','G3'),
                 ('spatial_order_same_parameters','C2','P2'),('spatial_order_same_parameters','C3','P3')]
    definitions += [('reference','S0',name) for name in study['screening']]
    definitions += [('middle_information',name+'_zero_mid',name) for name in ('F3','P3','G3')]
    pairs=[]
    for factor,before,after in definitions:
        if before not in groups or after not in groups:continue
        seeds=sorted(set(r['seed'] for r in groups[before])&set(r['seed'] for r in groups[after]))
        a=mean_seed_scores([r['scores'] for r in groups[before] if r['seed'] in seeds])
        b=mean_seed_scores([r['scores'] for r in groups[after] if r['seed'] in seeds])
        pairs.append({'factor':factor,'before':before,'after':after,'seeds':seeds,
                      'metrics':{key:scene_bootstrap(a,b,key) for key in KEYS},
                      'distribution':benefit_distribution(a,b)})
    interactions=[]
    for label,terms in [('fusion_by_stage',{'P3':1,'F3':-1,'P2':-1,'F2':1}),
                        ('gate_by_stage',{'G3':1,'P3':-1,'G2':-1,'P2':1})]:
        if not set(terms)<=set(groups):continue
        seeds=sorted(set.intersection(*[{r['seed'] for r in groups[name]} for name in terms]))
        averaged={name:{int(x['source_index']):x for x in mean_seed_scores([
            r['scores'] for r in groups[name] if r['seed'] in seeds])} for name in terms}
        indices=list(averaged[next(iter(terms))]);zero=[];change=[]
        for index in indices:
            scene=averaged[next(iter(terms))][index]['scene']
            zero.append({'source_index':index,'scene':scene,**{k:0. for k in KEYS}})
            change.append({'source_index':index,'scene':scene,
                           **{k:sum(weight*averaged[name][index][k] for name,weight in terms.items()) for k in KEYS}})
        interactions.append({'name':label,'linear_contrast':terms,'seeds':seeds,
                             'metrics':{key:scene_bootstrap(zero,change,key) for key in KEYS}})
    destination=out/'analysis/factorial';destination.mkdir(parents=True,exist_ok=True)
    review={'summary':summary,'pairs':pairs,'interactions':interactions,'source_sha256':{
        (r['path']/'result.json').relative_to(ROOT).as_posix():sha(r['path']/'result.json')
        for rows in groups.values() for r in rows},
        'limits':'Repeated-validation exploratory study; no multiplicity correction; scene intervals and seed SD differ; partial rows are not a completed factorial.'}
    write_json(destination/'review.json',review)
    with (destination/'summary.csv').open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,['method','n_seeds',*KEYS,'abs_rel_sd','parameters'],extrasaction='ignore')
        writer.writeheader();writer.writerows(summary)
    if args.freeze:
        required=['S0',*study['screening']]
        for name in required:
            row=next(x for x in summary if x['method']==name)
            assert row['seeds']==study['confirmation_seeds'], ('Finish all three seeds',name)
        for name in study['controls']:assert name in groups
        ranked=sorted([x for x in summary if x['method'] in required],key=lambda x:x['abs_rel'])
        eligible=[x for x in ranked if x['abs_rel']<=ranked[0]['abs_rel']*(1+study['selection']['relative_tie'])]
        eligible.sort(key=lambda x:(x['parameters'],x['abs_rel']))
        winner=eligible[0]
        if len(study['candidates'][winner['method']]['branches'])==3:
            zero=winner['method']+'_zero_mid'
            assert zero in groups and sorted(r['seed'] for r in groups[zero])==study['confirmation_seeds'], 'Run selected zero-middle control before final freeze'
        source=[*[ROOT/name for name in ('configs/fusion_lora_study.json',study['parent_config'],study['split'],
                    'src/p3_depth/data/nyuv2.py','src/p3_depth/models/runtime.py',
                    'src/p3_depth/evaluation/course.py','src/p3_depth/evaluation/depth.py',
                    'src/p3_depth/evaluation/boundary.py','src/p3_depth/evaluation/rgb_depth_edges.py')],
                *sorted((ROOT/'src/p3_depth/residual').glob('*.py')),
                *sorted((ROOT/'src/p3_depth/residual_inputs').glob('*.py')),
                *[ROOT/'src/p3_depth/adaptation_study'/name for name in ('model.py','data.py','residual_training.py','common.py')]]
        for name,runs in groups.items():
            if name=='S0':continue
            for run in runs:
                provenance=json.loads((run['path']/'provenance.json').read_text())['source_sha256']
                for path in source:
                    key=path.relative_to(ROOT).as_posix()
                    assert provenance[key]==sha(path),('Mixed numerical sources before Residual freeze',name,run['seed'],key)
        decision={'status':'frozen_before_LoRA_search_and_new_test','at':now(),'method':winner['method'],
                  'epochs':int(np.median(winner['best_epochs'])),'schedule_epochs':80,'seeds':study['confirmation_seeds'],
                  'validation_abs_rel':winner['abs_rel'],'selection':study['selection'],
                  'evidence_sha256':sha(destination/'review.json'),
                  'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in source},
                  'statement':'Selected within a fixed candidate set; see paired uncertainty before claiming a win.'}
        path=out/'residual_frozen.json';assert not path.exists(),'Do not overwrite selection'
        write_json(path,decision);print('FROZEN '+json.dumps(decision),flush=True)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
