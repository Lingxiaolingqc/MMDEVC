"""Matched follow-up and frozen-test comparisons from retained per-image scores."""
from pathlib import Path
import argparse
import csv
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_residual_study import read_csv, scene_bootstrap, mean_seed_scores, benefit_distribution

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from p3_depth.residual.records import write_json, sha

METRICS = ('abs_rel', 'rmse_m', 'delta1', 'boundary_f1', 'boundary_band_abs_rel')


def collect(directory, section):
    records = []
    if section == 'test':
        frozen = json.loads((directory/'frozen_selection.json').read_text(encoding='utf-8'))
        planned = {x['run_id']:x for x in frozen['refits']}
        for path in sorted((directory/'final_test').glob('*/result.json')):
            result = json.loads(path.read_text(encoding='utf-8'))
            name = path.parent.name
            if result['status'] != 'complete': continue
            info = planned.get(name, {})
            records.append({'run':name, 'method':info.get('method', name), 'seed':info.get('seed'),
                            'metrics':result['metrics'], 'scores':read_csv(path.parent/'per_image.csv'),
                            'source':path, 'phase':'test'})
        expected = set(frozen['baseline_methods']) | set(planned)
        if {x['run'] for x in records} != expected:
            raise ValueError('All frozen test methods must finish before the final comparison')
        return records
    for parent in ('runs', 'followups'):
        for path in sorted((directory/parent).glob('*/result.json')):
            result = json.loads(path.read_text(encoding='utf-8'))
            if result.get('smoke') or result['status'] != 'complete' or not result.get('validation'): continue
            config = json.loads((path.parent/'config.json').read_text(encoding='utf-8'))
            job = config.get('job', {})
            records.append({'run':result['run'], 'method':config['candidate']['name'], 'seed':config['seed'],
                            'metrics':result['validation'], 'scores':read_csv(path.parent/'validation/per_image.csv'),
                            'source':path, 'phase':result.get('phase', 'screening'), 'config':config,
                            'fit_images':len(config['fit_ids']), 'feature_hw':config['feature_hw'],
                            'updates':result.get('updates_attempted'), 'exposures':result.get('image_exposures'),
                            'seconds':result['elapsed_seconds'], 'parameters':result['parameters']})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/residual_study.json')
    parser.add_argument('--section', choices=['combination', 'data_size', 'resolution', 'test'], required=True)
    args = parser.parse_args()
    study = json.loads((ROOT/args.config).read_text(encoding='utf-8'))
    directory = ROOT/study['output']; records = collect(directory, args.section)
    pairs = []; chosen = []; groups = {}
    by_run = {x['run']:x for x in records}

    def pair(reference, candidate, reason):
        if reference is None or candidate is None: return
        pairs.append({'reference':reference['run'], 'candidate':candidate['run'], 'comparison':reason,
                      'metrics':[scene_bootstrap(reference['scores'], candidate['scores'], key) for key in METRICS],
                      'benefit_distribution':benefit_distribution(reference['scores'], candidate['scores'])})

    if args.section == 'combination':
        chosen = [x for x in records if x['phase'] in ('combination', 'ablation') or x['method'] in ('R0', 'frozen_features', 'unet')]
        for row in chosen:
            suffix = '_seed'+str(row['seed'])
            if row['method'] == 'C1_features_unet':
                for reference in ('frozen_features', 'unet'):
                    pair(by_run.get(reference+suffix), row, 'same-seed single-component change')
            if row['method'] in ('C1_zero_features', 'C1_no_rgb'):
                pair(by_run.get('C1_features_unet'+suffix), row, 'same-seed component removal')
    elif args.section == 'data_size':
        chosen = [x for x in records if x['phase'] == 'data_size']
        trained = list(chosen)
        for row in trained:
            references = [x for x in chosen if x['method']=='R0' and x['fit_images']==row['fit_images'] and x['seed']==row['seed']]
            if row['method'] != 'R0' and references: pair(references[0], row, 'same fitting subset, validation, updates and exposures')
        for count in sorted({x['fit_images'] for x in trained}):
            path = directory/'data_size_bases'/f'n{count}'/'result.json'
            scalar = json.loads(path.read_text(encoding='utf-8'))
            row = {'run':f'B1_n{count}', 'method':'B1_scale', 'seed':None, 'fit_images':count,
                   'metrics':scalar['metrics'], 'scores':read_csv(ROOT/scalar['per_image_file']), 'source':path}
            chosen.append(row)
            for candidate in trained:
                if candidate['fit_images']==count: pair(row, candidate, 'same labeled subset; scalar reference is not neural matched compute')
            path = directory/'bases/B0/metrics.json'
            chosen.append({'run':f'B0_context_n{count}', 'method':'B0_frozen', 'seed':None, 'fit_images':count,
                           'metrics':json.loads(path.read_text(encoding='utf-8')), 'scores':read_csv(path.parent/'per_image.csv'), 'source':path})
    elif args.section == 'resolution':
        chosen = [x for x in records if x['phase']=='resolution']
        for row in list(chosen):
            references = [x for x in records if x['phase'] in ('screening','combination','ablation')
                          and x['method']==row['method'] and x['seed']==row['seed'] and x['feature_hw']==[240,320]]
            if len(references)!=1: raise ValueError('Resolution control needs one matched 240 reference')
            chosen.extend(x for x in references if x not in chosen)
            pair(references[0], row, 'same seed, method and epoch protocol; spatial resolution changes compute')
    else:
        chosen = records
    if not chosen: raise ValueError('No completed requested experiments')
    for row in chosen:
        key = row['method']
        if args.section=='data_size': key += '_n'+str(row['fit_images'])
        if args.section=='resolution': key += '_h'+str(row['feature_hw'][0])
        groups.setdefault(key, []).append(row)
    summary = []; averaged = {}
    for name, rows in groups.items():
        values = {'method':name, 'runs':[x['run'] for x in rows], 'seeds':[x['seed'] for x in rows]}
        for key in METRICS:
            v = np.array([x['metrics'][key] for x in rows])
            values[key] = float(v.mean()); values[key+'_sd'] = float(v.std(ddof=1)) if len(v)>1 else None
        summary.append(values)
        averaged[name] = {'run':name, 'scores':mean_seed_scores([x['scores'] for x in rows])}
    if args.section=='combination' and len(groups.get('C1_features_unet', []))==3:
        for name in ('frozen_features','unet'):
            if len(groups.get(name, []))==3: pair(averaged[name], averaged['C1_features_unet'], 'three-seed mean per-image scores; not a prediction ensemble')
    if args.section=='test':
        for name in averaged:
            if name!='B0': pair(averaged['B0'], averaged[name], 'frozen full test; mean scores across seeds where applicable')
            if name not in ('B0','B1') and 'B1' in averaged: pair(averaged['B1'], averaged[name], 'increment beyond development-fitted global scale')
            if name not in ('B0','B1','B2','B3','R0') and 'R0' in averaged: pair(averaged['R0'], averaged[name], 'increment beyond basic residual')
            if name not in ('B0','B1','B2','B3'):
                for reference in ('B2','B3'):
                    if reference in averaged: pair(averaged[reference], averaged[name], 'comparison with predeclared flip and scale-plus-flip controls')
    output = directory/'analysis'/args.section; output.mkdir(parents=True, exist_ok=True)
    review = {'section':args.section, 'split':'test' if args.section=='test' else 'validation',
              'summary':summary, 'paired_comparisons':pairs,
              'source_sha256':{str(x['source'].relative_to(ROOT)):sha(x['source']) for x in chosen},
              'limitations':['scene-bootstrap intervals are exploratory and unadjusted for multiple comparisons',
                            'scene uncertainty and training-seed SD are distinct',
                            'single-seed controls cannot establish seed robustness',
                            'averaging per-image scores is not prediction ensembling'],
              'test_history':study['protocol']['test_history']}
    write_json(output/'review.json', review)
    fields = ['method','abs_rel','abs_rel_sd','rmse_m','delta1','boundary_f1','boundary_band_abs_rel']
    with (output/'summary.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fields,extrasaction='ignore');writer.writeheader();writer.writerows(summary)
    text = [f'# Residual {args.section}: measured comparisons','',f"Split: {review['split']}.",'',
            '| Method | AbsRel | Seed SD | RMSE m | delta1 | Boundary F1 | Band AbsRel |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for row in summary:
        sd='n/a' if row['abs_rel_sd'] is None else f"{row['abs_rel_sd']:.6f}"
        text.append(f"| {row['method']} | {row['abs_rel']:.6f} | {sd} | {row['rmse_m']:.6f} | {row['delta1']:.6f} | {row['boundary_f1']:.6f} | {row['boundary_band_abs_rel']:.6f} |")
    text += ['', 'Paired per-image comparisons, scene-cluster intervals and benefit distributions are in review.json.',
             'No qualitative claim is implied by this generated table; visual inspection is recorded separately.']
    (output/'review.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    if args.section=='data_size':
        fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
        for method in sorted(set(x['method'] for x in chosen)):
            rows=sorted([x for x in chosen if x['method']==method],key=lambda x:x['fit_images'])
            for ax,key in zip(axes,('abs_rel','boundary_f1')):
                ax.plot([x['fit_images'] for x in rows],[x['metrics'][key] for x in rows],'-o',label=method)
        for ax,key in zip(axes,('AbsRel (lower better)','Boundary F1 (higher better)')):
            ax.set(xlabel='Fitting images; nested whole scenes',ylabel=key);ax.legend()
        fig.suptitle('Fixed 2000 updates / 16000 image exposures; final iterate; seed 320112')
    else:
        fig,axes=plt.subplots(1,2,figsize=(11,max(4,len(summary)*.35)),constrained_layout=True)
        for ax,key in zip(axes,('abs_rel','boundary_f1')):
            ax.barh([x['method'] for x in summary],[x[key] for x in summary],xerr=[x[key+'_sd'] or 0 for x in summary])
            ax.set_xlabel(key+'; error bars = sample seed SD');ax.invert_yaxis()
    fig.savefig(output/'comparison.png',dpi=160);fig.savefig(output/'comparison.pdf');plt.close(fig)
    print(f'PASS {args.section}: {len(chosen)} scored entries, {len(pairs)} matched comparisons: {output}')


if __name__=='__main__': main()
