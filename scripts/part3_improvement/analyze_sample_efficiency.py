"""Summarize completed sample-efficiency runs; incomplete cells stay explicit."""
import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from p3_depth.sample_efficiency.protocol import load, outdir, jobs, name, check_freeze
from p3_depth.residual.records import sha, write_json, now

METRICS = ['abs_rel', 'rmse_m', 'delta1', 'boundary_f1', 'boundary_band_abs_rel']
METHODS = ['R', 'L', 'J', 'S']


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def csv_rows(path):
    with path.open(encoding='utf8', newline='') as stream:
        return list(csv.DictReader(stream))


def save_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values):
    values = [x for x in values if x is not None]
    return {'n': len(values), 'mean': float(np.mean(values)) if values else None,
            'sd': float(np.std(values, ddof=1)) if len(values) > 1 else None}


def load_completed(study, cfg):
    complete, pending = {}, []
    expected = set(read(study / 'subsets.json')['validation_ids'])
    for job in jobs(cfg):
        run = study / 'runs' / name(job)
        if not (run / 'result.json').exists():
            pending.append(name(job))
            continue
        result = read(run / 'result.json')
        assert result['status'] == 'complete' and result['job'] == job
        assert result['freeze_sha256'] == sha(study / 'frozen.json')
        for filename, digest in result['checkpoint_sha256'].items():
            assert sha(run / filename) == digest, (run, filename)
        result['history'] = read(run / 'history.json')
        result['attempt_events'] = [json.loads(line) for line in (run / 'attempts.jsonl').read_text(encoding='utf8').splitlines() if line]
        result['rows'] = {}
        for checkpoint in ['best', 'final']:
            rows = csv_rows(run / (checkpoint + '_validation') / 'per_image.csv')
            assert len(rows) == len(expected) and {int(r['source_index']) for r in rows} == expected
            result['rows'][checkpoint] = {r['sample_id']: r for r in rows}
            for metric in METRICS:
                values = [float(r[metric]) for r in rows if r[metric] != '']
                assert abs(float(np.mean(values)) - result[checkpoint][metric]) < 1e-10
        complete[name(job)] = result
    return complete, pending


def time_at(history, step):
    if step is None:
        return None
    return next(h['elapsed_seconds'] for h in history if h['step'] == step)


def flatten(result):
    history = result['history']
    cv = result['convergence']
    row = {**result['job'], 'parameters': result['parameters'], 'stop_reason': result['stop_reason'],
           'best_step': cv['best_step'], 'stop_step': cv['stop_step'], 'stable_step': cv['stable_step'],
           'target_0_10_step': cv['targets']['0.1'], 'target_0_09_step': cv['targets']['0.09'],
           'exposures': result['exposures'], 'effective_epochs': result['exposures'] / result['job']['count'],
           'successful_updates': result['successful_updates'], 'amp_skips': result['amp_skips'],
           'best_seconds': time_at(history, cv['best_step']), 'stable_seconds': time_at(history, cv['stable_step']),
           'stop_seconds': history[-1]['elapsed_seconds'], 'total_seconds': result['elapsed_seconds'],
           'parent_lora_seconds': result['parent_lora_seconds'],
           'total_with_parent_seconds': result['elapsed_seconds'] + result['parent_lora_seconds'],
           'parent_lora_steps': result['parent_lora_attempted_updates'],
           'total_with_parent_steps': result['attempted_updates'] + result['parent_lora_attempted_updates'],
           'best_with_parent_steps': cv['best_step'] + result['parent_lora_attempted_updates'],
           'best_with_parent_seconds': time_at(history, cv['best_step']) + result['parent_lora_seconds'],
           'stable_with_parent_steps': None if cv['stable_step'] is None else cv['stable_step'] + result['parent_lora_attempted_updates'],
           'stable_with_parent_seconds': None if cv['stable_step'] is None else time_at(history, cv['stable_step']) + result['parent_lora_seconds'],
           'pure_train_seconds': sum(h.get('training_seconds', 0) for h in history),
           'selection_validation_seconds': sum(h['validation_seconds'] for h in history),
           'detailed_scoring_seconds': result['detailed_scoring_seconds'],
           'peak_allocated_mib': result['peak_allocated_mib']}
    for checkpoint in ['best', 'final']:
        row.update({checkpoint + '_' + k: result[checkpoint][k] for k in METRICS})
    row['final_minus_best_abs_rel'] = result['final']['abs_rel'] - result['best']['abs_rel']
    row['final_relative_degradation'] = row['final_minus_best_abs_rel'] / result['best']['abs_rel']
    after = [h for h in history if h['step'] > cv['best_step']]
    row['post_best_evaluations'] = len(after)
    row['post_best_max_abs_rel'] = max((h['validation']['abs_rel'] for h in after), default=None)
    around = [h['train_loss'] for h in history if max(1, cv['best_step'] - 200) <= h['step'] <= cv['best_step']]
    tail = [h['train_loss'] for h in history[-3:] if 'train_loss' in h]
    row['late_loss_down_val_up_indicator'] = bool(around and tail and np.median(tail) < np.median(around)
                                                 and row['final_relative_degradation'] > .01)
    events = result['attempt_events']
    durations = [(datetime.fromisoformat(e['at'])-datetime.fromisoformat(e['started'])).total_seconds()
                 for e in events if e['event'] in ['complete', 'failed'] and 'started' in e]
    row['recorded_attempt_wall_seconds'] = sum(durations)
    row['attempts_started'] = sum(e['event'] == 'started' for e in events)
    row['attempts_with_recorded_end'] = len(durations)
    row['attempt_wall_time_complete'] = row['attempts_started'] == len(durations)
    # A three-point target window starts before it can be confirmed online.
    for threshold, prefix in [('0.1','target_0_10'),('0.09','target_0_09')]:
        first = cv['targets'][threshold]
        confirm = None
        if first is not None:
            index = next(i for i,h in enumerate(history) if h['step']==first)
            confirm = history[index+2]['step']
        row[prefix+'_confirmed_step'] = confirm
        row[prefix+'_confirmed_seconds'] = time_at(history,confirm)
    return row


def pair_stats(candidate, reference, bootstrap=2000):
    """Average paired seeds per image, then resample scenes retaining image weights."""
    assert len(candidate) == len(reference) > 0
    ids = sorted(candidate[0])
    assert all(set(r) == set(ids) for r in candidate + reference)
    groups = defaultdict(list)
    for i, sample in enumerate(ids):
        scene = candidate[0][sample]['scene']
        assert all(r[sample]['scene'] == scene for r in candidate + reference)
        groups[scene].append(i)
    result = {}
    for metric in METRICS:
        if any(r[sample][metric] == '' for r in candidate + reference for sample in ids):
            result[metric] = {'status': 'undefined values; not silently dropped'}
            continue
        differences = np.array([[float(a[s][metric]) - float(b[s][metric]) for s in ids]
                                for a, b in zip(candidate, reference)])
        average = differences.mean(axis=0)
        sums = np.array([average[idx].sum() for idx in groups.values()])
        sizes = np.array([len(idx) for idx in groups.values()])
        rng = np.random.default_rng(320119)
        draws = rng.integers(0, len(groups), (bootstrap, len(groups)))
        samples = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
        result[metric] = {'candidate_minus_reference': float(average.mean()),
                          'paired_seed_differences': differences.mean(axis=1).tolist(),
                          'paired_seed_summary': mean_sd(differences.mean(axis=1).tolist()),
                          'scene_bootstrap_95_ci': np.quantile(samples, [.025, .975]).tolist()}
        lower_is_better = metric not in ['delta1','boundary_f1']
        improvement = -average if lower_is_better else average
        result[metric]['paired_image_distribution'] = {
            'images_improved':int((improvement>0).sum()), 'images_worsened':int((improvement<0).sum()),
            'images_exactly_tied':int((improvement==0).sum()), 'median_signed_difference':float(np.median(average)),
            'selection_rule':'rank paired image-macro differences averaged across matched seeds; descriptive, never used for training',
            'largest_improvement_cases':[{'sample_id':ids[i],'scene':candidate[0][ids[i]]['scene'],'signed_difference':float(average[i])}
                                         for i in [i for i in np.argsort(-improvement) if improvement[i]>0][:3]],
            'largest_regression_cases':[{'sample_id':ids[i],'scene':candidate[0][ids[i]]['scene'],'signed_difference':float(average[i])}
                                        for i in [i for i in np.argsort(improvement) if improvement[i]<0][:3]]}
    return {'seeds': len(candidate), 'scenes': len(groups), 'images': len(ids), 'metrics': result,
            'bootstrap': {'replicates': bootstrap, 'seed': 320119,
                          'estimand': 'image-macro paired difference, averaged across available paired training seeds',
                          'uncertainty': 'conditional scene sampling; training-seed variability reported separately',
                          'multiplicity': 'descriptive intervals, no multiple-comparison correction'}}


def comparisons(complete, baseline, cfg):
    output = []
    for count in cfg['counts']:
        for candidate, reference in [('R', 'B0'), ('L', 'B0'), ('J', 'B0'), ('S', 'B0'),
                                     ('L', 'R'), ('J', 'R'), ('J', 'L'), ('S', 'L'), ('S', 'J')]:
            seeds = [seed for seed in cfg['seeds'] if f'{candidate}_n{count}_seed{seed}' in complete
                     and (reference == 'B0' or f'{reference}_n{count}_seed{seed}' in complete)]
            if not seeds:
                continue
            for checkpoint in ['best', 'final']:
                a = [complete[f'{candidate}_n{count}_seed{s}']['rows'][checkpoint] for s in seeds]
                b = [baseline if reference == 'B0' else complete[f'{reference}_n{count}_seed{s}']['rows'][checkpoint]
                     for s in seeds]
                output.append({'count': count, 'candidate': candidate, 'reference': reference,
                               'checkpoint': checkpoint, 'seed_ids': seeds, 'complete_three_seeds': len(seeds) == 3,
                               **pair_stats(a, b)})
    # Marginal gain uses adjacent nested subset sizes, irrespective of execution order.
    increments = []
    first = cfg['counts'][0]
    for method in METHODS:
        seeds = [s for s in cfg['seeds'] if f'{method}_n{first}_seed{s}' in complete]
        if not seeds: continue
        for checkpoint in ['best','final']:
            a = [complete[f'{method}_n{first}_seed{s}']['rows'][checkpoint] for s in seeds]
            increments.append({'method':method, 'lower_count':0, 'higher_count':first,
                               'checkpoint':checkpoint, 'seed_ids':seeds, 'additional_images':first,
                               'zero_shot_reference':'one shared baseline, not independent seed measurements',
                               **pair_stats(a,[baseline]*len(seeds))})
    for low, high in zip(cfg['counts'], cfg['counts'][1:]):
        for method in METHODS:
            seeds = [s for s in cfg['seeds'] if f'{method}_n{low}_seed{s}' in complete and f'{method}_n{high}_seed{s}' in complete]
            if not seeds:
                continue
            for checkpoint in ['best', 'final']:
                a = [complete[f'{method}_n{high}_seed{s}']['rows'][checkpoint] for s in seeds]
                b = [complete[f'{method}_n{low}_seed{s}']['rows'][checkpoint] for s in seeds]
                increments.append({'method': method, 'lower_count': low, 'higher_count': high,
                                   'checkpoint': checkpoint, 'seed_ids': seeds, 'additional_images': high-low,
                                   **pair_stats(a, b)})
    for entry in increments:
        for metric, statistic in entry['metrics'].items():
            if 'candidate_minus_reference' in statistic:
                statistic['signed_change_per_100_added_images'] = statistic['candidate_minus_reference'] * 100 / entry['additional_images']
    return output, increments


def checkpoint_drift(complete, cfg):
    """Paired descriptive differences at stopping, relative to the AbsRel-selected best."""
    output = []
    for count in cfg['counts']:
        for method in METHODS:
            seeds = [seed for seed in cfg['seeds'] if f'{method}_n{count}_seed{seed}' in complete]
            if not seeds:
                continue
            selected = [complete[f'{method}_n{count}_seed{seed}'] for seed in seeds]
            output.append({'count': count, 'method': method, 'seed_ids': seeds,
                           'complete_three_seeds': len(seeds) == 3,
                           'candidate': 'final', 'reference': 'best',
                           'interpretation': 'Validation-selected best; descriptive endpoint drift, not an independent estimate of checkpoint-selection benefit. No boundary metrics were monitored between these endpoints.',
                           **pair_stats([r['rows']['final'] for r in selected], [r['rows']['best'] for r in selected])})
    return output


def plots(out, complete, rows, baseline, cfg, phase='Validation'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(zip(METHODS, ['#2171b5', '#d95f0e', '#238b45', '#756bb1']))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
    keys = METRICS + ['total_with_parent_seconds']
    for ax, metric in zip(axes.flat, keys):
        for method in METHODS:
            counts, means, sds = [], [], []
            for count in cfg['counts']:
                selected = [r for r in rows if r['method'] == method and r['count'] == count]
                if not selected:
                    continue
                key = 'best_' + metric if metric in METRICS else metric
                stats = mean_sd([r[key] for r in selected])
                counts.append(count); means.append(stats['mean']); sds.append(stats['sd'] or 0)
                ax.annotate(str(stats['n']), (count, stats['mean']), xytext=(0, 5), textcoords='offset points', fontsize=7)
            if counts:
                ax.errorbar(counts, means, yerr=sds, marker='o', label=method, color=colors[method], capsize=3)
        if metric in METRICS:
            ax.axhline(baseline[metric], color='black', ls='--', lw=.8, label='B0 (shared)')
        ax.set_title(metric); ax.set_xlabel('Unique fitting images'); ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(phase+': available completed seeds (labels = n); bars = sample SD. R timing excludes shared feature cache.')
    fig.savefig(out / 'sample_efficiency.png', dpi=160); plt.close(fig)
    for count in cfg['counts']:
        selected = [r for r in complete.values() if r['job']['count'] == count]
        if not selected:
            continue
        fig, axes = plt.subplots(2, 4, figsize=(16, 7), layout='constrained')
        for col, method in enumerate(METHODS):
            if not any(r['job']['method'] == method for r in selected):
                for row in range(2):
                    axes[row, col].text(.5, .5, method+'\nNo completed runs', ha='center', va='center', transform=axes[row, col].transAxes)
                    axes[row, col].set_axis_off()
                continue
            for result in selected:
                if result['job']['method'] != method:
                    continue
                h = result['history']; label = str(result['job']['seed'])
                axes[0, col].plot([r['step'] for r in h], [r['validation']['abs_rel'] for r in h], label=label)
                training = [r for r in h if 'train_loss' in r]
                axes[1, col].plot([r['step'] for r in training], [r['train_loss'] for r in training], label=label)
            axes[0, col].axhline(baseline['abs_rel'], color='black', ls='--', lw=.7)
            for row in range(2):
                axes[row, col].set_title(method + (' validation' if row == 0 else ' online train loss'))
                axes[row, col].set_xlabel('Attempted updates'); axes[row, col].grid(alpha=.2)
                if axes[row, col].get_legend_handles_labels()[0]: axes[row, col].legend(fontsize=7)
        fig.suptitle(f'{count} fitting images; full histories including post-best behavior')
        fig.savefig(out / f'curves_n{count}.png', dpi=150); plt.close(fig)


def render_fixed(out, study, complete, cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from p3_depth.sample_efficiency.data import Pack
    pack = Pack(ROOT)
    destination = out / 'fixed_cases'; destination.mkdir(exist_ok=True)
    seed = cfg['seeds'][0]  # Fixed before training; never pick the best-looking seed.
    meta = {'seed': seed, 'selection': 'predeclared sample IDs; first predeclared seed, no cherry-picking', 'files': []}
    try:
        for count in cfg['counts']:
            available = [m for m in METHODS if f'{m}_n{count}_seed{seed}' in complete]
            if not available:
                continue
            predictions = {m: np.load(study / 'runs' / f'{m}_n{count}_seed{seed}' / 'best_validation' / 'fixed_predictions.npz')
                           for m in available}
            try:
                for sample in cfg['fixed_visual_ids']:
                    i = pack.positions([sample-1])[0]
                    truth, mask = pack['truth'][i], pack['mask'][i]
                    rgb = pack.file['images'][sample-1].transpose(2, 1, 0)
                    methods = {'B0': pack['original'][i], **{m: predictions[m][str(sample)] for m in available}}
                    fig, axes = plt.subplots(2, len(methods)+1, figsize=(3*(len(methods)+1), 5.2), layout='constrained')
                    axes[0, 0].imshow(rgb); axes[0, 0].set_title('RGB')
                    axes[1, 0].imshow(np.ma.array(truth, mask=~mask), vmin=.001, vmax=10, cmap='viridis'); axes[1, 0].set_title('GT depth [m]')
                    for col, (method, pred) in enumerate(methods.items(), 1):
                        pred = np.clip(pred, .001, 10)
                        axes[0, col].imshow(pred, vmin=.001, vmax=10, cmap='viridis'); axes[0, col].set_title(method + ' depth [m]')
                        error = np.abs(pred-truth) / np.maximum(truth, .001)
                        axes[1, col].imshow(np.ma.array(error, mask=~mask), vmin=0, vmax=.5, cmap='magma')
                        axes[1, col].set_title(f'AbsRel={error[mask].mean():.4f}')
                    for ax in axes.flat: ax.axis('off')
                    fig.suptitle(f'NYUv2 {sample:04d}; n={count}, seed={seed}, best; depth 0–10m, relative error 0–0.5')
                    filename = f'n{count}_{"".join(available)}_nyuv2_{sample:04d}.png'
                    fig.savefig(destination / filename, dpi=150); plt.close(fig); meta['files'].append(filename)
                    edge_filename = filename.replace('.png','_edges.png')
                    render_boundary_case(destination/edge_filename, rgb, truth, mask, methods,
                                         f'NYUv2 {sample:04d}; n={count}, seed={seed}, best')
                    meta['files'].append(edge_filename)
            finally:
                for value in predictions.values(): value.close()
        write_json(destination / 'index.json', meta)
    finally: pack.close()


def render_incremental(out, study, complete, cfg):
    """Compare adjacent completed data counts using the fixed first seed and saved arrays."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from p3_depth.sample_efficiency.data import Pack
    pack = Pack(ROOT)
    destination = out / 'incremental_cases'; destination.mkdir(exist_ok=True)
    seed = cfg['seeds'][0]
    meta = {'seed': seed, 'selection': 'predeclared samples and first seed; adjacent sorted completed counts',
            'checkpoint': 'best', 'files': [], 'prediction_sources': {}}
    try:
        for method in METHODS:
            counts = sorted(n for n in cfg['counts'] if f'{method}_n{n}_seed{seed}' in complete)
            for lower, higher in zip(counts, counts[1:]):
                paths = {n: study/'runs'/f'{method}_n{n}_seed{seed}'/'best_validation'/'fixed_predictions.npz'
                         for n in [lower, higher]}
                predictions = {n: np.load(path) for n, path in paths.items()}
                meta['prediction_sources'].update({path.relative_to(study).as_posix(): sha(path) for path in paths.values()})
                try:
                    for sample in cfg['fixed_visual_ids']:
                        i = pack.positions([sample-1])[0]
                        truth, mask = pack['truth'][i], pack['mask'][i]
                        rgb = pack.file['images'][sample-1].transpose(2, 1, 0)
                        methods = {'B0': pack['original'][i],
                                   **{f'{method} n={n}': predictions[n][str(sample)] for n in [lower, higher]}}
                        fig, axes = plt.subplots(2, 4, figsize=(12, 5.2), layout='constrained')
                        axes[0, 0].imshow(rgb); axes[0, 0].set_title('RGB')
                        axes[1, 0].imshow(np.ma.array(truth, mask=~mask), vmin=.001, vmax=10, cmap='viridis')
                        axes[1, 0].set_title('GT depth [m]')
                        for col, (label, pred) in enumerate(methods.items(), 1):
                            pred = np.clip(pred, .001, 10)
                            axes[0, col].imshow(pred, vmin=.001, vmax=10, cmap='viridis'); axes[0, col].set_title(label)
                            error = np.abs(pred-truth) / np.maximum(truth, .001)
                            axes[1, col].imshow(np.ma.array(error, mask=~mask), vmin=0, vmax=.5, cmap='magma')
                            axes[1, col].set_title(f'AbsRel={error[mask].mean():.4f}')
                        for ax in axes.flat: ax.axis('off')
                        title = f'NYUv2 {sample:04d}; {method} {lower} to {higher} images; seed={seed}, best'
                        fig.suptitle(title+'; depth 0-10m, relative error 0-0.5')
                        filename = f'{method}_n{lower}_to_n{higher}_nyuv2_{sample:04d}.png'
                        fig.savefig(destination/filename, dpi=150); plt.close(fig); meta['files'].append(filename)
                        edge_filename = filename.replace('.png', '_edges.png')
                        render_boundary_case(destination/edge_filename, rgb, truth, mask, methods, title)
                        meta['files'].append(edge_filename)
                finally:
                    for prediction in predictions.values(): prediction.close()
        write_json(destination/'index.json', meta)
    finally:
        pack.close()


def render_checkpoints(out, study, complete, cfg, label):
    """Inspect a completed run's predeclared best/final outputs without inference."""
    if label not in complete:
        raise ValueError(f'Checkpoint comparison requires a completed study run: {label}')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from p3_depth.sample_efficiency.data import Pack
    pack = Pack(ROOT)
    destination = out/'checkpoint_cases'/label; destination.mkdir(parents=True, exist_ok=True)
    paths = {k: study/'runs'/label/(k+'_validation')/'fixed_predictions.npz' for k in ['best', 'final']}
    predictions = {k: np.load(path) for k, path in paths.items()}
    meta = {'job': complete[label]['job'], 'selection': 'predeclared fixed sample IDs; both saved best and final',
            'files': [], 'prediction_sources': {p.relative_to(study).as_posix(): sha(p) for p in paths.values()}}
    try:
        for sample in cfg['fixed_visual_ids']:
            i = pack.positions([sample-1])[0]
            truth, mask = pack['truth'][i], pack['mask'][i]
            rgb = pack.file['images'][sample-1].transpose(2, 1, 0)
            methods = {'B0': pack['original'][i], **{k: v[str(sample)] for k, v in predictions.items()}}
            fig, axes = plt.subplots(2, 4, figsize=(12, 5.2), layout='constrained')
            axes[0, 0].imshow(rgb); axes[0, 0].set_title('RGB')
            axes[1, 0].imshow(np.ma.array(truth, mask=~mask), vmin=.001, vmax=10, cmap='viridis')
            axes[1, 0].set_title('GT depth [m]')
            for col, (checkpoint, pred) in enumerate(methods.items(), 1):
                pred = np.clip(pred, .001, 10)
                axes[0, col].imshow(pred, vmin=.001, vmax=10, cmap='viridis'); axes[0, col].set_title(checkpoint)
                error = np.abs(pred-truth) / np.maximum(truth, .001)
                axes[1, col].imshow(np.ma.array(error, mask=~mask), vmin=0, vmax=.5, cmap='magma')
                axes[1, col].set_title(f'AbsRel={error[mask].mean():.4f}')
            for ax in axes.flat: ax.axis('off')
            title = f'NYUv2 {sample:04d}; {label}; best versus final'
            fig.suptitle(title+'; depth 0-10m, relative error 0-0.5')
            filename = f'nyuv2_{sample:04d}.png'
            fig.savefig(destination/filename, dpi=150); plt.close(fig); meta['files'].append(filename)
            edge_filename = filename.replace('.png', '_edges.png')
            render_boundary_case(destination/edge_filename, rgb, truth, mask, methods, title)
            meta['files'].append(edge_filename)
        write_json(destination/'index.json', meta)
    finally:
        for prediction in predictions.values(): prediction.close()
        pack.close()


def render_boundary_case(path, rgb, truth, mask, methods, title):
    """Saved predictions only; exact same boundary definition as numerical scoring."""
    import matplotlib.pyplot as plt
    from p3_depth.evaluation.boundary import boundary_masks, evaluate_boundary
    from p3_depth.evaluation.rgb_depth_edges import near_edges
    _,_,gt_edges,_=boundary_masks(truth,truth,mask,mask)
    background=rgb.astype(np.float32)/255*.4
    fig,axes=plt.subplots(1,len(methods)+1,figsize=(3*(len(methods)+1),3.5),layout='constrained')
    gt_view=background.copy();gt_view[gt_edges]=[1,1,1]
    axes[0].imshow(gt_view);axes[0].set_title('GT discontinuities (white)')
    for ax,(method,pred) in zip(axes[1:],methods.items()):
        clipped=np.clip(pred,.001,10);valid=np.isfinite(clipped)&(clipped>0)
        _,_,gt,pred_edges=boundary_masks(clipped,truth,mask,valid)
        matched=pred_edges&near_edges(gt,2)
        missed=gt&~near_edges(pred_edges,2)
        wrong=pred_edges&~near_edges(gt,2)
        display=background.copy();display[missed]=[1,.15,.15];display[matched]=[0,1,1];display[wrong]=[1,.85,0]
        score=evaluate_boundary(clipped,truth,mask,valid)['f1']
        ax.imshow(display);ax.set_title(method+f' F1={score:.3f}' if score is not None else method+' F1 undefined')
    for ax in axes:ax.axis('off')
    fig.suptitle(title+'\nRed: missed GT; cyan: matched prediction; yellow: unmatched prediction (2px tolerance)',fontsize=10)
    fig.savefig(path,dpi=180);plt.close(fig)


def convergence_plots(out, rows, baseline, cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(zip(METHODS,['#2171b5','#d95f0e','#238b45','#756bb1']))
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    items=[('best_step','Validation best'),('stop_step','Actual stop'),('stable_step','Stable at end'),
           ('target_0_10_confirmed_step','AbsRel <= 0.10: third qualifying evaluation'),('target_0_09_confirmed_step','AbsRel <= 0.09: third qualifying evaluation'),
           ('total_with_parent_steps','Total updates including parent L')]
    for ax,(key,title) in zip(axes.flat,items):
        plotted=False
        for method in METHODS:
            points=[]
            for count in cfg['counts']:
                selected=[r for r in rows if r['count']==count and r['method']==method]
                observed=[r[key] for r in selected if r[key] is not None]
                if not observed: continue
                s=mean_sd(observed);points.append((count,s['mean'],s['sd'] or 0))
                ax.annotate(f'{len(observed)}/{len(selected)}',(count,s['mean']),xytext=(0,6),textcoords='offset points',fontsize=7)
            if points:
                x,y,error=zip(*points);ax.errorbar(x,y,yerr=error,marker='o',label=method,color=colors[method],capsize=3);plotted=True
        ax.set_title(title);ax.set_xlabel('Unique fitting images');ax.set_ylabel('Attempted updates');ax.grid(alpha=.2)
        if plotted:
            ax.legend(fontsize=7)
            ax.set_ylim(bottom=0)
        else:
            ax.text(.5,.5,'Not observed in completed runs',ha='center',va='center',transform=ax.transAxes)
            ax.set_axis_off()
    fig.suptitle('Operational convergence: labels = attained/completed seeds; missing targets are not zero. S own-stage steps unless total is stated.')
    fig.savefig(out/'convergence_steps.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,metric in zip(axes,['abs_rel','boundary_f1','boundary_band_abs_rel']):
        for method in METHODS:
            for only_stable in [False,True]:
                points=[]
                for count in cfg['counts']:
                    selected=[r for r in rows if r['count']==count and r['method']==method and (not only_stable or r['stable_step'] is not None)]
                    if selected:
                        values=[r['final_'+metric] for r in selected];points.append((count,float(np.mean(values))))
                if points:
                    x,y=zip(*points);ax.plot(x,y,marker='s' if only_stable else 'o',ls='--' if only_stable else '-',color=colors[method],
                                           label=method+(' stable subset' if only_stable else ' all final'))
        ax.axhline(baseline[metric],color='black',ls=':',lw=.8,label='B0');ax.set_title(metric);ax.set_xlabel('Unique fitting images');ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.suptitle('Stopping-model performance; stable subset is conditional and may be empty')
    fig.savefig(out/'stopping_performance.png',dpi=150);plt.close(fig)


def export_records(study, complete):
    """Publish compact numerical evidence; leave weights, arrays and logs local."""
    destination = ROOT / 'results/sample_efficiency_records_20261127'
    destination.mkdir(exist_ok=True)
    relatives = ['frozen.json', 'source.zip', 'subsets.json', 'data_audit.json', 'preflight.json',
                 'command_failures.jsonl', 'environment.json', 'queue.json', 'queue_failures.jsonl',
                 'failures.jsonl', 'analysis_implementation_checks.json', 'resource_recording_update.json', 'shared_cache_costs.json', 'baseline/identity.json', 'baseline/metrics.json', 'baseline/per_image.csv',
                 'analysis/summary.json', 'analysis/per_seed.csv', 'analysis/paired_comparisons.json',
                 'analysis/incremental_gains.json', 'analysis/checkpoint_drift.json', 'analysis/incremental_cases/index.json']
    for label in complete:
        relatives += [f'runs/{label}/{f}' for f in ['result.json', 'config.json', 'history.json',
                      'attempts.jsonl', 'events.jsonl', 'startup_resources.jsonl', 'best_validation/metrics.json', 'best_validation/per_image.csv',
                      'final_validation/metrics.json', 'final_validation/per_image.csv']]
    relatives += [p.relative_to(study).as_posix() for p in (study/'analysis/visual_reviews').glob('*.json')]
    relatives += [p.relative_to(study).as_posix() for p in (study/'analysis/checkpoint_cases').glob('*/index.json')]
    for relative in relatives:
        source = study/relative
        if not source.exists(): continue
        target = destination/relative; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        assert sha(target) == sha(source)
    write_json(destination/'manifest.json', {'at':now(), 'completed_training_stages':len(complete), 'required':72,
               'status':'development_in_progress' if len(complete)<72 else 'development_complete_test_and_review_status_separate',
               'files':{p.relative_to(destination).as_posix():sha(p) for p in sorted(destination.rglob('*'))
                        if p.is_file() and p.name!='manifest.json'}})


def analyze_test(study, cfg, render=False, export=False):
    """Test results retain validation-selected identity; never select by test score."""
    frozen = read(study/'test_frozen.json')
    assert frozen['source_freeze_sha256'] == sha(study/'frozen.json')
    expected_ids = set(frozen['test_ids']); assert len(expected_ids) == 654
    development, _ = load_completed(study, cfg); assert len(development) == 72

    def result(label, checkpoint):
        folder = study/'final_test'/label/checkpoint
        saved = read(folder/'result.json')
        assert saved['status'] == 'complete' and saved['identity']['test_freeze_sha256'] == sha(study/'test_frozen.json')
        assert saved['identity']['checkpoint'] == checkpoint
        for filename, digest in saved['files'].items(): assert sha(folder/filename) == digest
        rows = csv_rows(folder/'per_image.csv')
        assert len(rows) == 654 and {int(r['source_index']) for r in rows} == expected_ids
        for metric in METRICS:
            values = [float(r[metric]) for r in rows if r[metric] != '']
            assert abs(np.mean(values)-saved['metrics'][metric]) < 1e-10
        return saved, {r['sample_id']:r for r in rows}

    base, base_rows = result('B0', 'original')
    complete, rows, pending = {}, [], []
    for job in jobs(cfg):
        label = name(job)
        if not all((study/'final_test'/label/c/'result.json').exists() for c in ['best','final']):
            pending.append(label); continue
        saved = {c:result(label,c) for c in ['best','final']}
        dev = development[label]
        for checkpoint,(record,_) in saved.items():
            assert record['identity']['job'] == job
            assert record['identity']['checkpoint_sha256'] == dev['checkpoint_sha256'][checkpoint+'.pt']
        complete[label] = {'job':job, 'rows':{c:v[1] for c,v in saved.items()}}
        row = {**job, 'validation_selected_best_step':dev['convergence']['best_step'],
               'stopping_step':dev['convergence']['stop_step'],
               'parameters':dev['parameters'],
               'total_with_parent_seconds':dev['elapsed_seconds']+dev['parent_lora_seconds']}
        row.update({c+'_'+metric:saved[c][0]['metrics'][metric] for c in saved for metric in METRICS})
        row['stopping_minus_validation_selected_test_abs_rel'] = row['final_abs_rel']-row['best_abs_rel']
        rows.append(row)
    groups = []
    for count in cfg['counts']:
        for method in METHODS:
            selected = [r for r in rows if r['count']==count and r['method']==method]
            groups.append({'count':count, 'method':method, 'completed_seeds':len(selected),
                           'statistics':{c+'_'+k:mean_sd([r[c+'_'+k] for r in selected]) for c in ['best','final'] for k in METRICS}
                                        if selected else {}})
    paired, increments = comparisons(complete, base_rows, cfg)
    out = study/'test_analysis'; out.mkdir(exist_ok=True)
    report = {'at':now(), 'phase':'official_654_test', 'complete_training_jobs_evaluated':len(complete),
              'required':72, 'pending':pending, 'baseline':base['metrics'], 'groups':groups,
              'test_freeze_sha256':sha(study/'test_frozen.json'), 'analysis_script_sha256':sha(Path(__file__)),
              'selection':'best means selected on validation; final means actual stopping state. Test never selects a checkpoint.',
              'test_history':frozen['test_history'],
              'scope':'All samples are from each declared fitting subset; no training on validation or all795 refit.'}
    write_json(out/'summary.json',report); save_csv(out/'per_seed.csv',rows)
    write_json(out/'paired_comparisons.json',paired); write_json(out/'incremental_gains.json',increments)
    if render:
        plots(out, {}, rows, base['metrics'], cfg, phase='Official test (validation-selected best)')
        render_test_cases(out, study, complete, cfg, frozen)
    if export:
        destination = ROOT/'results/sample_efficiency_records_20261127'
        relatives = ['test_frozen.json','test_complete.json','live_bridge.json']
        relatives += [p.relative_to(study).as_posix() for p in out.glob('*.json')]
        relatives += [p.relative_to(study).as_posix() for p in out.glob('*.csv')]
        for label in ['B0']+list(complete):
            for checkpoint in (['original'] if label=='B0' else ['best','final']):
                relatives += [f'final_test/{label}/{checkpoint}/{f}' for f in ['result.json','metrics.json','per_image.csv']]
        for relative in relatives:
            source = study/relative
            if source.exists():
                target=destination/relative; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(source.read_bytes())
        export_records(study, development)
    print(json.dumps({'test_jobs_completed':len(complete),'required':72,'output':str(out)}))


def render_test_cases(out, study, complete, cfg, frozen):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from p3_depth.data.nyuv2 import NYUv2Labeled
    seed=cfg['seeds'][0]; destination=out/'fixed_cases'; destination.mkdir(exist_ok=True)
    entries=[]
    with NYUv2Labeled(ROOT/cfg['data']) as data, np.load(study/'final_test/B0/original/fixed_predictions.npz') as baseline:
        for count in cfg['counts']:
            available=[m for m in METHODS if f'{m}_n{count}_seed{seed}' in complete]
            if not available: continue
            for checkpoint in ['best','final']:
                predictions={m:np.load(study/'final_test'/f'{m}_n{count}_seed{seed}'/checkpoint/'fixed_predictions.npz') for m in available}
                try:
                    for sample_id in frozen['fixed_test_preview_ids']:
                        sample=data.load(sample_id); mask=sample.valid_gt; truth=sample.depth_m
                        methods={'B0':baseline[str(sample_id)],**{m:predictions[m][str(sample_id)] for m in available}}
                        fig,axes=plt.subplots(2,len(methods)+1,figsize=(3*(len(methods)+1),5.2),layout='constrained')
                        axes[0,0].imshow(sample.rgb);axes[0,0].set_title('RGB')
                        axes[1,0].imshow(np.ma.array(truth,mask=~mask),vmin=.001,vmax=10,cmap='viridis');axes[1,0].set_title('GT depth [m]')
                        for col,(method,prediction) in enumerate(methods.items(),1):
                            pred=np.clip(prediction,.001,10); error=np.abs(pred-truth)/np.maximum(truth,.001)
                            axes[0,col].imshow(pred,vmin=.001,vmax=10,cmap='viridis');axes[0,col].set_title(method+' depth [m]')
                            axes[1,col].imshow(np.ma.array(error,mask=~mask),vmin=0,vmax=.5,cmap='magma');axes[1,col].set_title(f'AbsRel={error[mask].mean():.4f}')
                        for ax in axes.flat:ax.axis('off')
                        fig.suptitle(f'Official test NYUv2 {sample_id:04d}; n={count}, seed={seed}, {checkpoint}; depth 0–10m, error 0–0.5')
                        filename=f'n{count}_{"".join(available)}_{checkpoint}_nyuv2_{sample_id:04d}.png'
                        fig.savefig(destination/filename,dpi=150);plt.close(fig);entries.append(filename)
                finally:
                    for value in predictions.values():value.close()
    write_json(destination/'index.json',{'fixed_seed':seed,'sample_ids':frozen['fixed_test_preview_ids'],
                                      'selection':'Frozen first/middle/last official test IDs; first predefined seed; both checkpoint types',
                                      'files':entries})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render', action='store_true', help='Render fixed samples and curves from saved outputs (no inference)')
    parser.add_argument('--render-increments', action='store_true', help='Render adjacent data-count comparisons of the same fixed samples/seed (no inference)')
    parser.add_argument('--render-checkpoints', metavar='RUN_LABEL', help='Render saved best/final fixed samples for one completed development run (no inference)')
    parser.add_argument('--export-records', action='store_true', help='Copy verified compact records to results/ for version control')
    parser.add_argument('--test', action='store_true', help='Analyze frozen 654-image test outputs; never run inference or choose checkpoints')
    args = parser.parse_args()
    if args.test and (args.render_increments or args.render_checkpoints):
        parser.error('--render-increments and --render-checkpoints currently render development validation only')
    cfg = load(ROOT); check_freeze(ROOT); study = outdir(ROOT)
    if args.test:
        analyze_test(study,cfg,args.render,args.export_records)
        return
    out = study / 'analysis'; out.mkdir(exist_ok=True)
    complete, pending = load_completed(study, cfg)
    rows = [flatten(r) for r in complete.values()]
    lora_parameters = read(study/'preflight.json')['methods']['L']['trainable_parameters']
    for row in rows:
        row['inference_adaptation_parameters'] = row['parameters'] + (lora_parameters if row['method']=='S' else 0)
    by_job = {(r['method'], r['count'], r['seed']):r for r in rows}
    for row in rows:
        parent = by_job[('L',row['count'],row['seed'])] if row['method']=='S' else None
        row['attempt_wall_with_parent_seconds'] = row['recorded_attempt_wall_seconds'] + (parent['recorded_attempt_wall_seconds'] if parent else 0)
        row['attempt_wall_with_parent_complete'] = row['attempt_wall_time_complete'] and (parent['attempt_wall_time_complete'] if parent else True)
    groups = []
    for count in cfg['counts']:
        for method in METHODS:
            selected = [r for r in rows if r['count'] == count and r['method'] == method]
            groups.append({'count': count, 'method': method, 'completed_seeds': len(selected),
                           'complete_three_seeds': len(selected) == 3,
                           'stable_seeds': sum(r['stable_step'] is not None for r in selected),
                           'stable_final_metrics':{k:mean_sd([r['final_'+k] for r in selected if r['stable_step'] is not None]) for k in METRICS},
                           'statistics': {key: mean_sd([r[key] for r in selected]) for key in rows[0]
                                          if key not in ['method', 'count', 'seed', 'stop_reason']} if selected else {}})
    baseline = read(study / 'baseline' / 'metrics.json')
    base_rows = {r['sample_id']: r for r in csv_rows(study / 'baseline' / 'per_image.csv')}
    paired, increments = comparisons(complete, base_rows, cfg)
    feature_metadata = read(ROOT / cfg['features'] / 'metadata.json')
    cache_cost_record = study/'shared_cache_costs.json'
    report = {'at': now(), 'phase': 'development_validation', 'completed': len(complete), 'required': 72,
              'pending': pending, 'baseline': baseline, 'groups': groups,
              'freeze_sha256': sha(study / 'frozen.json'), 'analysis_script_sha256': sha(Path(__file__)),
              'shared_cache_cost': {'feature_extraction_seconds': feature_metadata['elapsed_seconds'],
                                    'scope': 'Historical shared 795-image seven-feature extraction; includes more branches than F3 and is not a dedicated R benchmark',
                                    'detailed_cost_record': 'shared_cache_costs.json' if cache_cost_record.exists() else None,
                                    'detailed_cost_record_sha256': sha(cache_cost_record) if cache_cost_record.exists() else None,
                                    'metadata_sha256': sha(ROOT / cfg['features'] / 'metadata.json')},
              'notes': ['Only completed runs enter summaries; no extrapolation to missing seeds.',
                        'Scene bootstrap uses seed-averaged image differences; seed SD is separate.',
                        'Online training loss and validation accuracy use different data; divergence is a diagnostic, not proof of overfitting.',
                        'R uses cached frozen foundation features; R step timing excludes precomputation, unlike online L/J/S.',
                        'Elapsed times on this laptop are observational; not isolated speed benchmarks.',
                        'Recorded attempt wall time includes ended failures and setup; missing end events are explicitly flagged. Resumed training-clock totals can omit work lost after the last checkpoint.',
                        'Targets/stability are predefined operational criteria, not mathematical convergence.',
                        'target_*_step is the start of the first qualifying three-evaluation window; target_*_confirmed_step is when its third evaluation finishes.',
                        '100 to 200 images increases scene coverage from 100 to 187 scenes as well as image count.',
                        'This file contains no new official test evaluation.']}
    write_json(out / 'summary.json', report); save_csv(out / 'per_seed.csv', rows)
    write_json(out / 'paired_comparisons.json', paired); write_json(out / 'incremental_gains.json', increments)
    write_json(out / 'checkpoint_drift.json', checkpoint_drift(complete, cfg))
    if args.render:
        plots(out, complete, rows, baseline, cfg); convergence_plots(out,rows,baseline,cfg); render_fixed(out, study, complete, cfg)
    if args.render_increments:
        render_incremental(out, study, complete, cfg)
    if args.render_checkpoints:
        render_checkpoints(out, study, complete, cfg, args.render_checkpoints)
    if args.export_records:
        export_records(study, complete)
    print(json.dumps({'completed': len(complete), 'required': 72, 'output': str(out)}))


if __name__ == '__main__':
    main()
