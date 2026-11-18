"""Paired final extension comparisons with all retained parent controls."""
import csv,json,sys
from pathlib import Path
import numpy as np
from analyze_residual_study import read_csv,scene_bootstrap,mean_seed_scores,benefit_distribution
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from p3_depth.residual_inputs.common import load
from p3_depth.residual_inputs.final import check_freeze
from p3_depth.residual.records import sha,write_json
KEYS=['abs_rel','rmse_m','delta1','boundary_f1','boundary_band_abs_rel']

def main():
    study,_=load(ROOT);frozen,_=check_freeze(ROOT,study);out=ROOT/study['output'];parent=ROOT/study['parent_output'];groups={};sources={}
    def add(name,path):
        result=json.loads((path/'result.json').read_text(encoding='utf8'));assert result['status']=='complete' and result['metrics']['images']==654
        for artifact,value in result['artifact_sha256'].items():assert sha(path/artifact)==value
        groups.setdefault(name,[]).append({'scores':read_csv(path/'per_image.csv'),'metrics':result['metrics'],'path':path})
        sources[(path/'result.json').relative_to(ROOT).as_posix()]=sha(path/'result.json')
    for name in ['B0','B1','B2','B3']:add(name,parent/'final_test'/name)
    for seed in study['confirmation_seeds']:
        add('R0',parent/'final_test'/f'final_R0_seed{seed}');add('S0',parent/'final_test'/f'final_frozen_features_seed{seed}')
    for name,choice in frozen['methods'].items():
        for seed in choice['seeds']:add(name,out/'final_test'/f'refit_{name}_seed{seed}')
    summary=[];averages={}
    for name,rows in groups.items():
        row={'method':name,'n_seeds':len(rows)}
        for key in KEYS:
            values=[r['metrics'][key] for r in rows];row[key]=float(np.mean(values));row[key+'_sd']=float(np.std(values,ddof=1)) if len(values)>1 else None
        summary.append(row);averages[name]=mean_seed_scores([r['scores'] for r in rows])
    pairs=[]
    for name in frozen['methods']:
        for ref in ['S0','B0','B1','B3','R0']:
            pairs.append({'candidate':name,'reference':ref,'metrics':{key:scene_bootstrap(averages[ref],averages[name],key) for key in KEYS},'benefit_distribution':benefit_distribution(averages[ref],averages[name])})
    directory=out/'analysis/test';directory.mkdir(parents=True,exist_ok=True)
    write_json(directory/'review.json',{'summary':summary,'pairs':pairs,'source_sha256':sources,'scope':'654 official test; mean scores across3 separately trained models, not ensemble; earlier test exposure disclosed; exploratory unadjusted scene-bootstrap intervals'})
    with (directory/'summary.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,['method','n_seeds',*KEYS,'abs_rel_sd'],extrasaction='ignore');w.writeheader();w.writerows(summary)
    print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
