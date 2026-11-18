"""Summarize measured controlled-study runs without treating missing runs as results."""
from pathlib import Path
import argparse
import csv
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from p3_depth.residual.records import write_json


def scene_bootstrap(reference, candidate, key='abs_rel'):
    left = {int(row['source_index']): row for row in reference}
    right = {int(row['source_index']): row for row in candidate}
    if left.keys() != right.keys():
        raise ValueError('Cannot compare unmatched image sets')
    groups = {}
    for index, row in left.items():
        if row['scene'] != right[index]['scene']:
            raise ValueError('Scene metadata mismatch')
        groups.setdefault(row['scene'], []).append(float(right[index][key])-float(row[key]))
    sums, counts = np.array([sum(x) for x in groups.values()]), np.array([len(x) for x in groups.values()])
    draws = np.random.default_rng(320118).integers(0, len(groups), (5000, len(groups)))
    replicates = sums[draws].sum(1)/counts[draws].sum(1)
    return {'metric':key,'candidate_minus_reference':float(sums.sum()/counts.sum()),
            'scene_cluster_bootstrap_ci95':np.quantile(replicates,[.025,.975]).tolist(),
            'scenes':len(groups),'images':len(left),'replicates':5000,
            'scope':'paired image difference with scene-cluster resampling; does not measure training-seed uncertainty'}


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def mean_seed_scores(runs):
    """Average per-image errors, never average predictions into an ensemble."""
    indexed=[{int(row['source_index']):row for row in run} for run in runs]
    if any(rows.keys()!=indexed[0].keys() for rows in indexed):raise ValueError('Seeds use different image sets')
    result=[]
    for index,first in indexed[0].items():
        if any(rows[index]['scene']!=first['scene'] for rows in indexed):raise ValueError('Seed scene metadata differs')
        result.append({'source_index':index,'scene':first['scene'],
                       **{key:float(np.mean([float(rows[index][key]) for rows in indexed]))
                          for key in ('abs_rel','rmse_m','delta1','boundary_f1','boundary_band_abs_rel')}})
    return result


def benefit_distribution(reference,candidate):
    left={int(row['source_index']):row for row in reference};right={int(row['source_index']):row for row in candidate}
    if left.keys()!=right.keys():raise ValueError('Unmatched benefit analysis')
    differences=np.array([float(right[i]['abs_rel'])-float(left[i]['abs_rel']) for i in left])
    gains=-differences[differences<0]
    return {'images':len(differences),'mean_change':float(differences.mean()),'median_change':float(np.median(differences)),
            'images_lower_absrel':int((differences<0).sum()),'images_higher_absrel':int((differences>0).sum()),
            'unchanged_images':int((differences==0).sum()),'change_quantiles':dict(zip(['min','q25','median','q75','max'],np.quantile(differences,[0,.25,.5,.75,1]).tolist())),
            'largest10_share_of_gross_improvement':float(np.sort(gains)[-10:].sum()/gains.sum()) if len(gains) else None,
            'scope':'paired image error changes; gross improvement denominator excludes worsened images; not a significance test'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/residual_study.json')
    args=parser.parse_args()
    study=json.loads((ROOT/args.config).read_text(encoding='utf-8'))
    directory=ROOT/study['output']
    output=directory/'analysis'
    output.mkdir(exist_ok=True)
    base=json.loads((directory/'bases/comparison.json').read_text(encoding='utf-8'))['validation']
    reference=read_csv(directory/'bases/B0/per_image.csv')
    rows=[]
    comparisons={}
    for label,values in base.items():
        rows.append({'method':label,'seed':None,'smoke':False,**values})
        if label!='B0':
            comparisons[label]=scene_bootstrap(reference,read_csv(directory/f'bases/{label}/per_image.csv'))
    completed=[]
    matched=[]
    records_by_run={}
    phases={}
    paths=sorted((directory/'runs').glob('*/result.json'))+sorted((directory/'followups').glob('*/result.json'))
    for path in paths:
        result=json.loads(path.read_text(encoding='utf-8'))
        if result['status']!='complete' or result['smoke'] or result.get('validation') is None:
            continue
        config=json.loads((path.parent/'config.json').read_text(encoding='utf-8'))
        label=result['run']
        history=json.loads((path.parent/'history.json').read_text(encoding='utf-8'))
        phases[label]=result.get('phase','screening_or_confirmation')
        rows.append({'method':label,'seed':config['seed'],'smoke':False,
                     'parameters':result['parameters'],'best_epoch':result.get('best_epoch',result.get('best_unit')),
                     'completed_epochs':history[-1]['epoch'],'elapsed_seconds':result['elapsed_seconds'],
                     'peak_allocated_mib':result['peak_allocated_mib'],**result['validation']})
        measured_rows=read_csv(path.parent/'validation/per_image.csv')
        records_by_run[label]=measured_rows
        comparisons[label]=scene_bootstrap(reference,measured_rows)
        own_base={'original':'B0','scale':'B1','flip':'B2','flip_scale':'B3'}[config['candidate']['base']]
        own_scores=read_csv(directory/f'bases/{own_base}/per_image.csv')
        base_comparison='global577_scale_reference' if phases[label]=='data_size' and own_base in ('B1','B3') else 'own_base'
        matched.append({'run':label,'comparison':base_comparison,'reference':own_base,
                        **scene_bootstrap(own_scores,measured_rows)})
        completed.append(label)
        fig,axes=plt.subplots(1,2,figsize=(10,3.6),constrained_layout=True)
        epochs=[x['epoch'] for x in history]
        axes[0].plot(epochs,[x['train_loss'] for x in history],label='Training objective')
        validation=[x for x in history if 'validation' in x]
        axes[0].plot([x['epoch'] for x in validation],[x['validation']['abs_rel'] for x in validation],label='Validation AbsRel')
        axes[0].set(xlabel='Epoch',ylabel='Loss / AbsRel',title=label)
        axes[0].legend()
        axes[1].plot([x['updates_attempted'] for x in validation],[x['validation']['abs_rel'] for x in validation])
        axes[1].set(xlabel='Attempted optimizer updates',ylabel='Validation AbsRel',title='Training-budget view')
        fig.savefig(output/(label+'_learning_curve.png'),dpi=160)
        plt.close(fig)
    for label,values in records_by_run.items():
        seed=label.rsplit('_seed',1)[1]
        ref_name='R0_seed'+seed
        if ref_name in records_by_run and label!=ref_name:
            for key in ('abs_rel','rmse_m','boundary_f1','boundary_band_abs_rel'):
                matched.append({'run':label,'comparison':('common_R0_same_seed_different_budget' if phases[label]=='data_size' else 'common_R0_same_seed'),'reference':ref_name,
                                **scene_bootstrap(records_by_run[ref_name],values,key)})
    seed_groups={}
    for row in rows:
        if row['seed'] is not None:
            seed_groups.setdefault(row['method'].rsplit('_seed',1)[0],[]).append(row)
    seed_summary=[]
    for method,group in seed_groups.items():
        seeds=sorted(int(x['seed']) for x in group)
        for metric in ('abs_rel','rmse_m','delta1','boundary_f1','boundary_band_abs_rel'):
            values=np.array([x[metric] for x in group])
            seed_summary.append({'method':method,'metric':metric,'seeds':seeds,'n_seeds':len(group),
                                 'mean':float(values.mean()),'std':float(values.std(ddof=1)) if len(group)>1 else None,
                                 'min':float(values.min()),'max':float(values.max()),
                                 'all_declared_seeds_complete':seeds==sorted(study['confirmation_seeds'])})
    mean_rows={}
    for method,group in seed_groups.items():
        if sorted(int(x['seed']) for x in group)==sorted(study['confirmation_seeds']):
            mean_rows[method]=mean_seed_scores([records_by_run[x['method']] for x in group])
    seed_averaged_comparisons=[]
    for method,values in mean_rows.items():
        for ref_name,ref in [('B0',reference),*([('R0',mean_rows['R0'])] if 'R0' in mean_rows and method!='R0' else [])]:
            for metric in ('abs_rel','rmse_m','boundary_f1','boundary_band_abs_rel'):
                seed_averaged_comparisons.append({'method':method,'reference':ref_name,'seeds':study['confirmation_seeds'],
                    'averaging':'per-image scores across trained seeds, not prediction ensemble',**scene_bootstrap(ref,values,metric)})
    distributions={}
    for label,values in records_by_run.items():
        ref_name='R0_seed'+label.rsplit('_seed',1)[1]
        if ref_name in records_by_run and label!=ref_name:
            distributions[label]={'reference':ref_name,**benefit_distribution(records_by_run[ref_name],values)}
    if seed_summary:
        with (output/'seed_summary.csv').open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(seed_summary[0]));writer.writeheader();writer.writerows(seed_summary)
    fields=['method','seed','images','abs_rel','rmse_m','delta1','boundary_f1','boundary_pooled_f1','boundary_band_abs_rel','nonboundary_abs_rel',
            'parameters','best_epoch','completed_epochs','elapsed_seconds','peak_allocated_mib']
    with (output/'comparison.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)
    write_json(output/'review.json',{'split':'validation','test_used':False,'completed_formal_training_runs':completed,
               'paired_comparisons_vs_B0':comparisons,'matched_controls':matched,'seed_summary':seed_summary,
               'three_seed_mean_score_comparisons':seed_averaged_comparisons,'benefit_distributions_vs_R0':distributions,
               'limitations':['screening uses one seed until explicitly confirmed','not final test results',
                              'baseline/architecture/input/output/loss causes must be separated',
                              'screening intervals are exploratory, not corrected for multiple comparisons']})
    if matched:
        with (output/'matched_controls.csv').open('w',encoding='utf-8',newline='') as stream:
            fields=['run','comparison','reference','metric','candidate_minus_reference','ci95_low','ci95_high','scenes','images']
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader()
            writer.writerows({**x,'ci95_low':x['scene_cluster_bootstrap_ci95'][0],'ci95_high':x['scene_cluster_bootstrap_ci95'][1]} for x in matched)
    text=['# Residual study: measured validation results','',
          'All scores below are measured on the fixed 218-image validation split. This is not final test evaluation.',
          '', '| Method | AbsRel ↓ | RMSE m ↓ | delta1 ↑ | Boundary macro F1 ↑ | Boundary-band AbsRel ↓ |',
          '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        text.append(f'| {row["method"]} | {row["abs_rel"]:.6f} | {row["rmse_m"]:.6f} | {row["delta1"]:.6f} | {row["boundary_f1"]:.6f} | {row["boundary_band_abs_rel"]:.6f} |')
    text+=['','B0=original; B1=fitting-only global scale; B2=25% restored flip blend; B3=blend plus fitting-only scale.',
           'The old bundle is not assumed effective. Compare Residual with its own base as well as B0.',
           'Paired scene-bootstrap intervals are in review.json. They do not replace seed confirmation.',
           'Qualitative inspection is pending unless explicitly recorded in a separate visual review.']
    (output/'review.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),constrained_layout=True)
    labels=list(base)
    for ax,key,title in [(axes[0],'abs_rel','Validation AbsRel (lower better)'),(axes[1],'boundary_f1','Boundary macro F1 (higher better)')]:
        ax.bar(labels,[base[k][key] for k in labels],color=['#777777','#247ba0','#e9a23b','#57a773'])
        ax.set_title(title)
    fig.savefig(output/'base_comparison.png',dpi=160);plt.close(fig)
    r0='R0_seed'+str(study['screening_seed'])
    if r0 in records_by_run:
        ref_values={x['source_index']:float(x['abs_rel']) for x in records_by_run[r0]}
        labels=[x for x in records_by_run if x!=r0 and x.endswith('_seed'+str(study['screening_seed']))]
        if labels:
            differences=[[float(x['abs_rel'])-ref_values[x['source_index']] for x in records_by_run[label]] for label in labels]
            fig,ax=plt.subplots(figsize=(10,max(3,len(labels)*.34)),constrained_layout=True)
            ax.boxplot(differences,vert=False,tick_labels=[x.rsplit('_seed',1)[0] for x in labels],
                       flierprops={'markersize':2,'alpha':.35})
            ax.axvline(0,color='black',ls='--',lw=1)
            ax.set(xlabel='Per-image AbsRel difference vs R0 (negative is better)',
                   title='Matched validation images; all outliers retained')
            fig.savefig(output/'paired_error_distributions.png',dpi=160)
            fig.savefig(output/'paired_error_distributions.pdf')
            plt.close(fig)
    print(f'PASS analyzed {len(base)} bases and {len(completed)} formal training runs: {output}',flush=True)


if __name__=='__main__':
    main()
