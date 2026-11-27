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
    return output, increments


def plots(out, complete, rows, baseline, cfg):
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
    fig.suptitle('Validation: available completed seeds (labels = n); bars = sample SD. R timing excludes shared feature cache.')
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
            finally:
                for value in predictions.values(): value.close()
        write_json(destination / 'index.json', meta)
    finally: pack.close()


def export_records(study, complete):
    """Publish compact numerical evidence; leave weights, arrays and logs local."""
    destination = ROOT / 'results/sample_efficiency_records_20261127'
    destination.mkdir(exist_ok=True)
    relatives = ['frozen.json', 'source.zip', 'subsets.json', 'data_audit.json', 'preflight.json',
                 'command_failures.jsonl', 'environment.json', 'queue.json', 'queue_failures.jsonl',
                 'failures.jsonl', 'baseline/identity.json', 'baseline/metrics.json', 'baseline/per_image.csv',
                 'analysis/summary.json', 'analysis/per_seed.csv', 'analysis/paired_comparisons.json',
                 'analysis/incremental_gains.json']
    for label in complete:
        relatives += [f'runs/{label}/{f}' for f in ['result.json', 'config.json', 'history.json',
                      'attempts.jsonl', 'events.jsonl', 'best_validation/metrics.json', 'best_validation/per_image.csv',
                      'final_validation/metrics.json', 'final_validation/per_image.csv']]
    relatives += [p.relative_to(study).as_posix() for p in (study/'analysis/visual_reviews').glob('*.json')]
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render', action='store_true', help='Render fixed samples and curves from saved outputs (no inference)')
    parser.add_argument('--export-records', action='store_true', help='Copy verified compact records to results/ for version control')
    args = parser.parse_args()
    cfg = load(ROOT); check_freeze(ROOT); study = outdir(ROOT)
    out = study / 'analysis'; out.mkdir(exist_ok=True)
    complete, pending = load_completed(study, cfg)
    rows = [flatten(r) for r in complete.values()]
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
                           'statistics': {key: mean_sd([r[key] for r in selected]) for key in rows[0]
                                          if key not in ['method', 'count', 'seed', 'stop_reason']} if selected else {}})
    baseline = read(study / 'baseline' / 'metrics.json')
    base_rows = {r['sample_id']: r for r in csv_rows(study / 'baseline' / 'per_image.csv')}
    paired, increments = comparisons(complete, base_rows, cfg)
    feature_metadata = read(ROOT / cfg['features'] / 'metadata.json')
    report = {'at': now(), 'phase': 'development_validation', 'completed': len(complete), 'required': 72,
              'pending': pending, 'baseline': baseline, 'groups': groups,
              'freeze_sha256': sha(study / 'frozen.json'), 'analysis_script_sha256': sha(Path(__file__)),
              'shared_cache_cost': {'feature_extraction_seconds': feature_metadata['elapsed_seconds'],
                                    'scope': 'Historical shared 795-image seven-feature extraction; includes more branches than F3 and is not a dedicated R benchmark',
                                    'metadata_sha256': sha(ROOT / cfg['features'] / 'metadata.json')},
              'notes': ['Only completed runs enter summaries; no extrapolation to missing seeds.',
                        'Scene bootstrap uses seed-averaged image differences; seed SD is separate.',
                        'Online training loss and validation accuracy use different data; divergence is a diagnostic, not proof of overfitting.',
                        'R uses cached frozen foundation features; R step timing excludes precomputation, unlike online L/J/S.',
                        'Elapsed times on this laptop are observational; not isolated speed benchmarks.',
                        'Recorded attempt wall time includes ended failures and setup; missing end events are explicitly flagged. Resumed training-clock totals can omit work lost after the last checkpoint.',
                        'Targets/stability are predefined operational criteria, not mathematical convergence.',
                        'This file contains no new official test evaluation.']}
    write_json(out / 'summary.json', report); save_csv(out / 'per_seed.csv', rows)
    write_json(out / 'paired_comparisons.json', paired); write_json(out / 'incremental_gains.json', increments)
    if args.render:
        plots(out, complete, rows, baseline, cfg); render_fixed(out, study, complete, cfg)
    if args.export_records:
        export_records(study, complete)
    print(json.dumps({'completed': len(complete), 'required': 72, 'output': str(out)}))


if __name__ == '__main__':
    main()
