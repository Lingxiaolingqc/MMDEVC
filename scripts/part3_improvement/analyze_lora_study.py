"""Bounded LoRA search summaries, validation shortlist and immutable selection."""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from p3_depth.adaptation_study.common import load,check_residual_freeze
from p3_depth.adaptation_study.lora_training import candidates
from p3_depth.residual.records import sha,write_json,now
from analyze_residual_study import read_csv,mean_seed_scores,scene_bootstrap,benefit_distribution
KEYS=['abs_rel','rmse_m','delta1','boundary_f1','boundary_band_abs_rel']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--select',action='store_true');parser.add_argument('--freeze',action='store_true')
    args=parser.parse_args();study,_=load(ROOT);out=ROOT/study['output']
    check_residual_freeze(ROOT,study)
    groups={}
    for path in sorted((out/'followups').glob('lora_*/result.json')):
        result=json.loads(path.read_text());cfg=json.loads((path.parent/'config.json').read_text())
        if result['smoke'] or result['phase'] not in ('search','confirm'):continue
        groups.setdefault(cfg['candidate']['name'],[]).append({'seed':cfg['seed'],'result':result,
            'path':path,'scores':read_csv(path.parent/'validation/per_image.csv')})
    rows=[];pairs=[]
    b0=read_csv(ROOT/study['parent_output']/'bases/B0/per_image.csv')
    for name,runs in groups.items():
        assert len({r['seed'] for r in runs})==len(runs)
        row={'method':name,'n_seeds':len(runs),'seeds':sorted(r['seed'] for r in runs),
             'parameters':runs[0]['result']['parameters'],'best_updates':[r['result']['best_unit'] for r in runs]}
        for key in KEYS:
            values=[r['result']['validation'][key] for r in runs]
            row[key]=float(np.mean(values));row[key+'_sd']=float(np.std(values,ddof=1)) if len(values)>1 else None
        rows.append(row);average=mean_seed_scores([r['scores'] for r in runs])
        pairs.append({'method':name,'reference':'original_UniDepth','metrics':{k:scene_bootstrap(b0,average,k) for k in KEYS},
                      'distribution':benefit_distribution(b0,average)})
    destination=out/'analysis/lora';destination.mkdir(parents=True,exist_ok=True)
    review={'summary':rows,'pairs':pairs,'source_sha256':{r['path'].relative_to(ROOT).as_posix():sha(r['path']) for runs in groups.values() for r in runs},
            'limits':'Bounded validation search; not a global optimum; mean scores are not an ensemble; intervals do not correct multiplicity or incorporate training-seed uncertainty.'}
    write_json(destination/'review.json',review)
    with (destination/'summary.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,['method','n_seeds',*KEYS,'abs_rel_sd','parameters'],extrasaction='ignore');w.writeheader();w.writerows(rows)
    if args.select:
        assert set(groups)==set(candidates(study)), 'Complete all eight declared searches'
        screen=[]
        for name,runs in groups.items():
            run=next(r for r in runs if r['seed']==study['screening_seed'])
            screen.append((run['result']['validation']['abs_rel'],name))
        choice={'at':now(),'methods':[name for _,name in sorted(screen)[:2]],
                'rule':'two lowest same-seed validation AbsRel; append only the other two seeds',
                'screening_source_sha256':{r['path'].relative_to(ROOT).as_posix():sha(r['path']) for runs in groups.values() for r in runs if r['seed']==study['screening_seed']}}
        path=out/'lora_shortlist.json';assert not path.exists(),'Do not overwrite shortlist';write_json(path,choice)
        print('SHORTLIST '+json.dumps(choice),flush=True)
    if args.freeze:
        shortlist=json.loads((out/'lora_shortlist.json').read_text())['methods']
        eligible=[r for r in rows if r['method'] in shortlist]
        assert len(eligible)==2 and all(r['seeds']==study['confirmation_seeds'] for r in eligible)
        best=min(r['abs_rel'] for r in eligible)
        tied=[r for r in eligible if r['abs_rel']<=best*1.002]
        winner=min(tied,key=lambda r:(r['parameters'],r['abs_rel']))
        residual_freeze=json.loads((out/'residual_frozen.json').read_text())
        source=list(dict.fromkeys([*[ROOT/name for name in residual_freeze['source_sha256']],
                                   *sorted((ROOT/'src/p3_depth/adaptation_study').glob('*.py'))]))
        for name,runs in groups.items():
            initial_hashes=[]
            for run in runs:
                provenance=json.loads((run['path'].parent/'provenance.json').read_text())['source_sha256']
                config=json.loads((run['path'].parent/'config.json').read_text())
                assert config['adapter_initialization_seed']==run['seed']
                initial_hashes.append(config['initial_adapter_sha256'])
                assert run['result']['frozen_parameters_unchanged']
                for path in source:
                    key=path.relative_to(ROOT).as_posix()
                    assert provenance[key]==sha(path),('Mixed numerical sources before LoRA freeze',name,run['seed'],key)
            assert len(initial_hashes)==len(set(initial_hashes)),('Repeated seeds used identical adapter initialization',name)
        decision={'status':'frozen_before_size_comparison_and_new_test','at':now(),'method':winner['method'],
                  'candidate':candidates(study)[winner['method']],'updates':int(np.median(winner['best_updates'])),
                  'schedule_steps':study['lora']['search_steps'],'seeds':study['confirmation_seeds'],
                  'validation_abs_rel':winner['abs_rel'],'evidence_sha256':sha(destination/'review.json'),
                  'source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in source}}
        path=out/'lora_frozen.json';assert not path.exists(),'Do not overwrite selection';write_json(path,decision)
        print('FROZEN '+json.dumps(decision),flush=True)
    print(json.dumps(rows),flush=True)


if __name__=='__main__':main()
