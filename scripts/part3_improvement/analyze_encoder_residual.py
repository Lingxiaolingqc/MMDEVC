"""Audit completed encoder comparisons, paired statistics, full curves and fixed cases."""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from p3_depth.encoder_residual.common import load,check_freeze,reference_path
from p3_depth.residual.records import sha,write_json,native_path,now
from analyze_residual_study import read_csv,mean_seed_scores,scene_bootstrap,benefit_distribution
KEYS=['abs_rel','rmse_m','delta1','boundary_f1','boundary_band_abs_rel']


def collect(path):
    cfg=json.loads((path/'config.json').read_text())
    result=json.loads((path/'result.json').read_text())
    assert result['status']=='complete' and result['validation']['images']==218
    for name,digest in (result.get('checkpoint_sha256') or {'best.pt':result['best_checkpoint_sha256']}).items():
        assert sha(path/name)==digest
    rows=read_csv(path/'validation/per_image.csv')
    assert len(rows)==218
    for key in KEYS:
        assert abs(np.mean([float(row[key]) for row in rows])-result['validation'][key])<1e-9
    return {'path':path,'cfg':cfg,'result':result,'rows':rows,'history':json.loads((path/'history.json').read_text())}


def summarize(study):
    check_freeze(ROOT,study)
    out=ROOT/study['output']; analysis=out/'analysis'; analysis.mkdir(exist_ok=True)
    groups={}; completed=[]
    for name in study['queue']:
        paths=[out/'runs'/f'{name}_seed{seed}' for seed in study['seeds']]
        if not all((p/'result.json').exists() for p in paths): continue
        completed.append(name); groups[name]=[collect(p) for p in paths]
        reference=study['candidates'][name]['reference']
        groups[reference]=[collect(reference_path(ROOT,study,reference,s)) for s in study['seeds']]
    if not completed: raise ValueError('No completed three-seed comparison; missing runs are not results')
    if 'F3' not in groups:
        groups['F3']=[collect(reference_path(ROOT,study,'F3',s)) for s in study['seeds']]
    averages={name:mean_seed_scores([r['rows'] for r in runs]) for name,runs in groups.items()}
    summaries=[]; trajectories=[]; seed_rows=[]
    for name,runs in groups.items():
        summary={'method':name,'parameters':runs[0]['result']['parameters'],'seeds':study['seeds']}
        for key in KEYS+['elapsed_seconds','peak_allocated_mib','refiner_seconds_per_image']:
            values=[r['result']['validation'][key] if key in KEYS else r['result'][key] for r in runs]
            summary[key]={'mean':float(np.mean(values)),'sd':float(np.std(values,ddof=1)),'values':values}
        summary['best_epochs']=[r['result'].get('best_epoch',r['result'].get('best_unit')) for r in runs]
        summaries.append(summary)
        for seed,run in zip(study['seeds'],runs):
            h=run['history']; val=[x for x in h if 'validation' in x]
            best=min(val,key=lambda x:x['validation']['abs_rel']); final=val[-1]
            epoch=lambda x:x.get('epoch',x.get('unit'))
            post=[x for x in val if epoch(x)>epoch(best)]
            trajectory={'method':name,'seed':seed,'best_epoch':epoch(best),'final_epoch':epoch(final),
                'best_abs_rel':best['validation']['abs_rel'],'final_abs_rel':final['validation']['abs_rel'],
                'final_minus_best':final['validation']['abs_rel']-best['validation']['abs_rel'],
                'post_best_max_abs_rel':max([x['validation']['abs_rel'] for x in post],default=None),
                'train_loss_at_best':best['train_loss'],'final_train_loss':h[-1]['train_loss'],
                'post_best_points':len(post),'amp_skipped_updates':run['result'].get('amp_skipped_updates'),
                'interpretation':'Online fitting loss and eval-mode validation differ; a gap alone is not proof of overfitting.'}
            trajectories.append(trajectory)
            seed_rows.append({'method':name,'seed':seed,'parameters':summary['parameters'],
                **{k:run['result']['validation'][k] for k in KEYS},'best_epoch':epoch(best),'final_epoch':epoch(final),
                'final_abs_rel':trajectory['final_abs_rel'],'elapsed_seconds':run['result']['elapsed_seconds'],
                'peak_allocated_mib':run['result']['peak_allocated_mib']})
            if name in completed:
                fig,axes=plt.subplots(1,2,figsize=(10,3.4),constrained_layout=True)
                axes[0].plot([epoch(x) for x in h],[x['train_loss'] for x in h]); axes[0].set_title('Fitting train-mode loss')
                axes[1].plot([epoch(x) for x in val],[x['validation']['abs_rel'] for x in val]); axes[1].set_title('Validation AbsRel')
                for ax in axes: ax.axvline(epoch(best),color='red',ls='--'); ax.set_xlabel('Epoch'); ax.grid(alpha=.2)
                fig.suptitle(f'{name}, seed {seed}; selected and final both retained')
                fig.savefig(analysis/f'curve_{name}_{seed}.png',dpi=135); plt.close(fig)
    paired={}
    for name in completed:
        reference=study['candidates'][name]['reference']
        paired[name]={'reference':reference,'mean_seed_metrics':{k:scene_bootstrap(averages[reference],averages[name],k) for k in KEYS},
            'per_seed':[{'seed':s,'metrics':{k:scene_bootstrap(a['rows'],b['rows'],k) for k in KEYS}}
                        for s,a,b in zip(study['seeds'],groups[reference],groups[name])],
            'distribution':benefit_distribution(averages[reference],averages[name])}
        left={int(r['source_index']):r for r in averages[reference]}; right={int(r['source_index']):r for r in averages[name]}
        changes=[right[i]['abs_rel']-left[i]['abs_rel'] for i in left]
        fig,axes=plt.subplots(1,2,figsize=(9,3.4),constrained_layout=True)
        axes[0].boxplot([[x['abs_rel'] for x in left.values()],[right[i]['abs_rel'] for i in left]],tick_labels=[reference,name])
        axes[0].set_ylabel('Per-image AbsRel; mean of seed errors')
        axes[1].hist(changes,bins=24); axes[1].axvline(0,color='red'); axes[1].set_xlabel(f'{name} - {reference}; negative = improvement')
        fig.savefig(analysis/f'paired_{name}.png',dpi=150); plt.close(fig)
    eligible=[r for r in summaries if r['method'] in completed+['F3']]
    best=min(r['abs_rel']['mean'] for r in eligible)
    tied=[r for r in eligible if r['abs_rel']['mean']<=best*(1+study['selection']['relative_tie'])]
    selected=min(tied,key=lambda r:(r['parameters'],r['abs_rel']['mean']))
    b0=json.loads((ROOT/study['reference_output']/'reference_B0/metrics.json').read_text())
    report={'at':now(),'completed':completed,'pending':[n for n in study['queue'] if n not in completed],
        'summary':summaries,'seeds':seed_rows,'paired':paired,'trajectories':trajectories,'B0':b0,
        'selected':selected['method'],'selection_provisional':len(completed)!=len(study['queue']),
        'selection_rule':study['selection'],
        'single_feature_source_interpretation':{'S0':'Decoder init = to_latents(conditioned_features[0]); the first aggregated encoder group selects block3 under depths=[3,6,9,12], stacking_fn=last; camera rays use other features too.',
            'ES0':'Normalized encoder block12 before stage aggregation and decoder processing.',
            'limit':'Changes tapped encoder depth as well as decoder conditioning; not a pure before/after-decoder transformation of the same tensor.'},
        'limits':['Validation reused: exploratory comparisons, no fresh test.',
        'Projection capacity, feature sources and constructed grids differ; whole-candidate attribution.',
        'Seed means are metrics of separate models, not prediction ensembling.',
        'Scene CIs exclude training-seed uncertainty and are not multiplicity corrected.',
        'Training time includes checkpoint IO and validation; historical runs are not controlled speed benchmarks.'],
        'design_sha256':sha(out/'design_frozen.json'),'analysis_source_sha256':sha(Path(__file__)),
        'records':{str((r['path']/'result.json').relative_to(ROOT)).replace('\\','/'):sha(r['path']/'result.json') for rs in groups.values() for r in rs}}
    write_json(analysis/'review.json',report)
    with (analysis/'per_seed.csv').open('w',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(seed_rows[0])); writer.writeheader(); writer.writerows(seed_rows)
    fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    for ax,key in zip(axes,['abs_rel','boundary_f1','boundary_band_abs_rel']):
        ax.errorbar(range(len(summaries)),[r[key]['mean'] for r in summaries],yerr=[r[key]['sd'] for r in summaries],fmt='o',capsize=3)
        ax.axhline(b0[key],ls='--',color='gray',label='B0'); ax.set_xticks(range(len(summaries)),[r['method'] for r in summaries],rotation=50)
        ax.set_title(key+'; mean +/- seed SD'); ax.grid(alpha=.2); ax.legend()
    fig.savefig(analysis/'comparison.png',dpi=160); fig.savefig(analysis/'comparison.pdf'); plt.close(fig)
    if len(completed)==len(study['queue']):
        selection={'at':now(),'method':selected['method'],'validation_abs_rel':selected['abs_rel']['mean'],
            'status':'selection_for_future_study_only','old_freezes_untouched':True,'design_sha256':report['design_sha256'],
            'rule':study['selection'],'no_refit_or_test_run':True}
        target=out/'residual_selection.json'
        if target.exists(): assert json.loads(target.read_text())['method']==selection['method']
        else: write_json(target,selection)
    return report


def export_records(study,report):
    out=ROOT/study['output']; target=ROOT/'results/encoder_residual_records_20261125'
    paths=[out/name for name in ('design_frozen.json','preflight.json','reference_audit.json','numerical_source.zip','queue.json','features/metadata.json','cleanup.jsonl','failures.jsonl','command_failures.jsonl','residual_selection.json')]
    paths+=list((out/'analysis').rglob('*'))
    for name in report['completed']:
        for seed in study['seeds']:
            run=out/'runs'/f'{name}_seed{seed}'
            paths+=list(run.glob('*.json'))+list(run.glob('*.jsonl'))
            paths+=list((run/'validation').glob('*'))+list((run/'final_validation').glob('*'))
    for src in paths:
        if not src.is_file(): continue
        dest=native_path(target/src.relative_to(out)); dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(src.read_bytes())
    text=['# Encoder-feature residual study','',
          'Fixed queue: ES0 -> EF2 -> EF3 -> ESS1 -> EP2 -> EP3; 577 fitting / 218 validation, three seeds.',
          'No test, all-795 refit, LoRA or sample-efficiency runs. See docs/encoder_residual_plan.md for adaptations.',
          '','Completed: '+', '.join(report['completed']), 'Pending: '+', '.join(report['pending']),
          '', '| Method | AbsRel mean +/- SD | RMSE | delta1 | Boundary F1 | Band AbsRel | Parameters |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for r in report['summary']:
        text.append(f"| {r['method']} | {r['abs_rel']['mean']:.6f} +/- {r['abs_rel']['sd']:.6f} | {r['rmse_m']['mean']:.6f} | {r['delta1']['mean']:.6f} | {r['boundary_f1']['mean']:.6f} | {r['boundary_band_abs_rel']['mean']:.6f} | {r['parameters']} |")
    text+=['','Current selection: '+report['selected']+(' (provisional; queue incomplete)' if report['selection_provisional'] else ' (fixed declared validation rule)'),
           '', 'Best/final weights and feature arrays remain local under '+study['output']+'.',
           'Full per-image metrics, curves, seed variation, paired scene CIs and best-to-final trajectories are in analysis/review.json.',
           'Read analysis/interpretation.md and analysis/qualitative_review.json for inspected S0/ES0 cases; figures are in analysis/visualizations/.',
           'Numerical source is preserved in numerical_source.zip. Historical decoder records are reused after reference_audit.json checks.',
           'S0 init uses the first decoder input branch (encoder block3 under official last stacking), then camera conditioning/to_latents; ES0 uses block12. This changes layer depth too, not just decoder conditioning.',
           'Costs recorded during different training runs are descriptive, not an isolated latency benchmark.',
           '', '## Run / resume', '',
           'Requires the existing development prediction cache, audited decoder reference runs and the verified',
           'StyleSketch disk_inputs cache (RGB/log-depth, GT, masks only). See config paths; these are reused read-only.',
           '', '```powershell',
           '$py = ".\\.venv-cuda\\Scripts\\python.exe"',
           '& $py -X utf8 scripts/part3_improvement/run_encoder_residual.py prepare',
           '& $py -X utf8 scripts/part3_improvement/run_encoder_residual.py preflight',
           '& $py -X utf8 scripts/part3_improvement/run_encoder_residual.py queue --through ES0 --resume',
           '# Continue the registered remaining methods in order:',
           '& $py -X utf8 scripts/part3_improvement/run_encoder_residual.py queue --resume',
           '# Analyze completed three-seed comparisons; run rendering while training is stopped:',
           '& $py -X utf8 scripts/part3_improvement/render_encoder_residual.py',
           '& $py -X utf8 scripts/part3_improvement/analyze_encoder_residual.py --export',
           '```', '', 'Completed runs are verified and reused. A failure stops the queue; --resume restores optimizer/RNG state.',
           'Do not run the standalone benchmark concurrently with training.']
    (target/'README.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    write_json(target/'record_manifest.json',{'at':now(),'sha256':{p.relative_to(target).as_posix():sha(p) for p in target.rglob('*') if p.is_file() and p.name!='record_manifest.json'}})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--export',action='store_true')
    args=parser.parse_args(); study=load(ROOT); report=summarize(study)
    if args.export: export_records(study,report)
    print(json.dumps({'completed':report['completed'],'pending':report['pending'],'selected':report['selected'],'provisional':report['selection_provisional'],
        'scores':[{k:r[k] for k in ('method','abs_rel','boundary_f1','boundary_band_abs_rel')} for r in report['summary']]}))
