"""Compare input-extension runs against same-seed frozen parent S0."""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
from analyze_residual_study import read_csv,scene_bootstrap,mean_seed_scores,benefit_distribution
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from p3_depth.residual_inputs.common import load
from p3_depth.residual.records import sha,write_json
KEYS=['abs_rel','rmse_m','delta1','boundary_f1','boundary_band_abs_rel']

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--section',choices=['screen','development'],default='screen');p.add_argument('--select',action='store_true')
    p.add_argument('--decide-confirmation',nargs='+',help='At most two preselected methods, each with all three seeds');a=p.parse_args()
    s,parent=load(ROOT);out=ROOT/s['output'];old=ROOT/parent['output'];records=[]
    for seed in s['confirmation_seeds']:
        d=old/'runs'/f'frozen_features_seed{seed}'
        records.append({'name':'S0','seed':seed,'scores':read_csv(d/'validation/per_image.csv'),'result':json.loads((d/'result.json').read_text()),'path':d/'result.json'})
    for path in sorted((out/'followups').glob('*/result.json')):
        result=json.loads(path.read_text());cfg=json.loads((path.parent/'config.json').read_text())
        if result['smoke'] or not result['validation']:continue
        if a.section=='screen' and result['phase']!='screen':continue
        records.append({'name':cfg['candidate']['name'],'seed':cfg['seed'],'scores':read_csv(path.parent/'validation/per_image.csv'),'result':result,'path':path})
    if a.section=='screen':records=[r for r in records if r['seed']==320112]
    groups={};summary=[];pairs=[]
    for r in records:groups.setdefault(r['name'],[]).append(r)
    for name,rs in groups.items():
        assert len({r['seed'] for r in rs})==len(rs),'Duplicate method/seed; specify a unique analysis scope'
        row={'method':name,'seeds':[r['seed'] for r in rs],'n_seeds':len(rs)}
        for key in KEYS:
            values=[r['result']['validation'][key] for r in rs];row[key]=float(np.mean(values));row[key+'_sd']=float(np.std(values,ddof=1)) if len(values)>1 else None
        row['parameters']=rs[0]['result']['parameters'];summary.append(row)
        if name=='S0':continue
        refs=[r for r in groups['S0'] if r['seed'] in row['seeds']]
        before=mean_seed_scores([r['scores'] for r in refs]);after=mean_seed_scores([r['scores'] for r in rs])
        pairs.append({'candidate':name,'reference':'S0','seeds':row['seeds'],'metrics':{key:scene_bootstrap(before,after,key) for key in KEYS},'distribution':benefit_distribution(before,after)})
    destination=out/'analysis'/a.section;destination.mkdir(parents=True,exist_ok=True)
    review={'summary':summary,'pairs':pairs,'source_sha256':{r['path'].relative_to(ROOT).as_posix():sha(r['path']) for r in records},'limits':'Exploratory repeated-validation comparisons; paired scene bootstrap is separate from training-seed uncertainty; mean scores are not an ensemble.'}
    write_json(destination/'review.json',review)
    with (destination/'summary.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,['method','n_seeds',*KEYS,'abs_rel_sd','parameters'],extrasaction='ignore');w.writeheader();w.writerows(summary)
    if a.select:
        assert a.section=='screen' and set(s['screening'])<=set(groups),'Finish all6 before selection'
        ref=next(x for x in summary if x['method']=='S0');rules=s['promotion'];decision=[]
        for row in summary:
            if row['method']=='S0':continue
            gain=(ref['abs_rel']-row['abs_rel'])/ref['abs_rel'];f1=row['boundary_f1']-ref['boundary_f1'];band=row['boundary_band_abs_rel']/ref['boundary_band_abs_rel']-1
            primary=gain>=rules['relative_absrel_gain_min'] and f1>=-rules['maximum_f1_drop'] and band<=rules['maximum_relative_band_error_increase']
            secondary=f1>=rules['boundary_only_secondary']['f1_gain_min'] and gain>=-rules['boundary_only_secondary']['maximum_relative_absrel_increase']
            decision.append({'method':row['method'],'relative_absrel_gain':gain,'f1_change':f1,'relative_band_error_change':band,'primary_eligible':primary,'secondary_eligible':secondary,'abs_rel':row['abs_rel']})
        primary=sorted([x for x in decision if x['primary_eligible']],key=lambda x:x['abs_rel'])
        selected=primary[:rules['max_confirmed_candidates']]
        if not selected:selected=sorted([x for x in decision if x['secondary_eligible']],key=lambda x:-x['f1_change'])[:rules['max_confirmed_candidates']]
        value={'rules':rules,'rows':decision,'initial_confirmation_shortlist':[x['method'] for x in selected],'conditional_three_stage':any(x['method']=='M2_two_stage' for x in primary),'evidence_sha256':sha(destination/'review.json')}
        path=out/'screen_selection.json'
        if path.exists():assert json.loads(path.read_text())==value,'Do not overwrite a prior decision'
        else:write_json(path,value)
        print(json.dumps(value,ensure_ascii=False),flush=True)
    if a.decide_confirmation:
        assert a.section=='development' and len(a.decide_confirmation)<=s['promotion']['max_confirmed_candidates']
        rules=s['promotion'];ref=next(x for x in summary if x['method']=='S0');decisions=[]
        for name in a.decide_confirmation:
            row=next(x for x in summary if x['method']==name)
            assert sorted(row['seeds'])==sorted(s['confirmation_seeds'])
            paired=next(x for x in pairs if x['candidate']==name)
            gain=(ref['abs_rel']-row['abs_rel'])/ref['abs_rel'];f1=row['boundary_f1']-ref['boundary_f1'];band=row['boundary_band_abs_rel']/ref['boundary_band_abs_rel']-1
            primary=gain>=rules['relative_absrel_gain_min'] and f1>=-rules['maximum_f1_drop'] and band<=rules['maximum_relative_band_error_increase']
            ci=paired['metrics']['abs_rel']['scene_cluster_bootstrap_ci95']
            decisions.append({'method':name,'relative_absrel_gain':gain,'f1_change':f1,'relative_band_error_change':band,
                              'mean_primary_eligible':primary,'absrel_scene_ci95':ci,'confirmed_main':primary and ci[1]<0})
        value={'status':'confirmation_complete','rows':decisions,'retained_methods':[x['method'] for x in decisions if x['confirmed_main']],
               'evidence_sha256':sha(destination/'review.json'),'limits':review['limits'],'stop_rule':'No new all795 refit or test if no retained method; retain prior S0 and all negative findings.'}
        path=out/'confirmation_decision.json'
        if path.exists():assert json.loads(path.read_text())==value
        else:write_json(path,value)
        print(json.dumps(value,ensure_ascii=False),flush=True)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for ax,key in zip(axes,['abs_rel','boundary_f1']):
        ax.bar([r['method'] for r in summary],[r[key] for r in summary]);ax.tick_params(axis='x',rotation=65);ax.set_title(key)
    fig.suptitle('Fixed validation; input extension; mean scores across available seeds')
    fig.savefig(destination/'comparison.png',dpi=150);fig.savefig(destination/'comparison.pdf');plt.close(fig)
    labels=[];differences=[]
    for name,rs in groups.items():
        if name=='S0':continue
        refs=[r for r in groups['S0'] if r['seed'] in [x['seed'] for x in rs]]
        before={int(x['source_index']):float(x['abs_rel']) for x in mean_seed_scores([r['scores'] for r in refs])}
        after=mean_seed_scores([r['scores'] for r in rs]);labels.append(name)
        differences.append([float(x['abs_rel'])-before[int(x['source_index'])] for x in after])
    if labels:
        fig,ax=plt.subplots(figsize=(9,max(3,len(labels)*.6)),constrained_layout=True)
        ax.boxplot(differences,vert=False,tick_labels=labels,flierprops={'markersize':2,'alpha':.4})
        ax.axvline(0,color='black',ls='--');ax.set(xlabel='Per-image AbsRel difference vs same-seed S0; negative is better',title='Fixed validation; all outliers retained')
        fig.savefig(destination/'paired_changes.png',dpi=160);fig.savefig(destination/'paired_changes.pdf');plt.close(fig)
    for r in records:
        if r['name']=='S0':continue
        history=json.loads((r['path'].parent/'history.json').read_text());valid=[x for x in history if 'validation' in x]
        fig,axes=plt.subplots(1,2,figsize=(10,3.4),constrained_layout=True)
        axes[0].plot([x['epoch'] for x in history],[x['train_loss'] for x in history],label='Fitting objective')
        axes[0].plot([x['epoch'] for x in valid],[x['validation']['abs_rel'] for x in valid],label='Validation AbsRel');axes[0].legend()
        axes[0].set(xlabel='Epoch',ylabel='AbsRel');axes[1].plot([x['updates_attempted'] for x in valid],[x['validation']['abs_rel'] for x in valid])
        axes[1].set(xlabel='Attempted updates',ylabel='Validation AbsRel');fig.suptitle(r['path'].parent.name)
        fig.savefig(destination/(r['path'].parent.name+'_curve.png'),dpi=140);plt.close(fig)
if __name__=='__main__':main()
