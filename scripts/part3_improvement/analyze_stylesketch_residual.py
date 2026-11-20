"""Validation-only complete-candidate comparison, curves and fixed visual cases."""
import argparse
import csv
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from analyze_residual_study import read_csv, mean_seed_scores, scene_bootstrap, benefit_distribution
from p3_depth.stylesketch_residual.common import load, check_freeze, audit_references
from p3_depth.stylesketch_residual.model import StyleSketchResidual
from p3_depth.adaptation_study.model import make_model
from p3_depth.residual.records import sha, write_json, now
from p3_depth.residual.model import corrected_depth
from p3_depth.residual_inputs.data import observable_inputs
from p3_depth.data.nyuv2 import NYUv2Labeled
from p3_depth.residual.final_evaluation import scoring_pack
from p3_depth.residual.evaluation import score_image
from p3_depth.evaluation.boundary import boundary_masks
KEYS = ['abs_rel', 'rmse_m', 'delta1', 'boundary_f1', 'boundary_band_abs_rel']


def collect(study):
    prior, out = ROOT/study['reference_output'], ROOT/study['output']
    groups = {}
    for name in ('F3', 'P3', 'G3', 'SS1'):
        groups[name] = []
        for seed in study['seeds']:
            run = out/'runs'/f'SS1_seed{seed}' if name == 'SS1' else prior/'followups'/f'factorial_{name}_seed{seed}'
            cfg = json.loads((run/'config.json').read_text())
            result = json.loads((run/'result.json').read_text())
            assert result['status'] == 'complete'
            assert all(sha(run/n) == h for n, h in result['checkpoint_sha256'].items())
            groups[name].append({'seed': seed, 'path': run, 'config': cfg, 'result': result,
                                 'scores': read_csv(run/'validation/per_image.csv'),
                                 'history': json.loads((run/'history.json').read_text())})
    return groups


def summarize(study, groups):
    out = ROOT/study['output']/'analysis'; out.mkdir(exist_ok=True)
    summary, curves = [], []
    averages = {name: mean_seed_scores([r['scores'] for r in runs]) for name, runs in groups.items()}
    for name, runs in groups.items():
        row = {'method': name, 'seeds': study['seeds'], 'parameters': runs[0]['result']['parameters']}
        for key in KEYS + ['elapsed_seconds', 'peak_allocated_mib', 'refiner_seconds_per_image']:
            values = [r['result']['validation'][key] if key in KEYS else r['result'][key] for r in runs]
            row[key] = {'mean': float(np.mean(values)), 'sd': float(np.std(values, ddof=1)), 'values': values}
        row['best_epochs'] = [r['result'].get('best_epoch', r['result'].get('best_unit')) for r in runs]
        row['completed_epochs'] = [r['result'].get('completed_epochs', r['result'].get('completed_units')) for r in runs]
        summary.append(row)
        for run in runs:
            h = run['history']; val = [x for x in h if 'validation' in x]
            last, best = val[-1], min(val, key=lambda x: x['validation']['abs_rel'])
            def epoch(x):
                return x.get('epoch', x.get('unit'))
            after = [x for x in val if epoch(x) > epoch(best)]
            curves.append({'method': name, 'seed': run['seed'], 'best_epoch': epoch(best), 'final_epoch': epoch(last),
                'best_abs_rel': best['validation']['abs_rel'], 'final_abs_rel': last['validation']['abs_rel'],
                'final_minus_best_abs_rel': last['validation']['abs_rel']-best['validation']['abs_rel'],
                'post_best_max_abs_rel': max([x['validation']['abs_rel'] for x in after], default=None),
                'train_loss_at_best': best['train_loss'], 'final_train_loss': h[-1]['train_loss'],
                'post_best_validation_points': len(after),
                'interpretation_boundary': 'Train is online train-mode loss; validation is eval-mode. A gap alone does not prove overfitting.'})
            fig, ax = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
            ax[0].plot([epoch(x) for x in h], [x['train_loss'] for x in h], label='Fitting train-mode loss')
            ax[1].plot([epoch(x) for x in val], [x['validation']['abs_rel'] for x in val], label='Validation eval-mode AbsRel')
            for a in ax:
                a.axvline(epoch(best), ls='--', color='red', label='Selected best')
                a.set_xlabel('Epoch'); a.legend(fontsize=8); a.grid(alpha=.25)
            fig.suptitle(f'{name} seed{run["seed"]}: full trajectory, not only best point')
            fig.savefig(out/f'curve_{name}_{run["seed"]}.png', dpi=145); plt.close(fig)
    b0 = ROOT/study['reference_output']/'reference_B0'
    baseline = json.loads((b0/'metrics.json').read_text())
    pairs = {'mean_seeds': {key: scene_bootstrap(averages['F3'], averages['SS1'], key) for key in KEYS},
             'per_seed': [{'seed': a['seed'], 'metrics': {key: scene_bootstrap(a['scores'], b['scores'], key) for key in KEYS}}
                          for a, b in zip(groups['F3'], groups['SS1'])],
             'distribution': benefit_distribution(averages['F3'], averages['SS1'])}
    eligible = [r for r in summary if r['method'] in study['selection_candidates']]
    best = min(r['abs_rel']['mean'] for r in eligible)
    tied = [r for r in eligible if r['abs_rel']['mean'] <= best*(1+study['selection']['relative_tie'])]
    selected = min(tied, key=lambda r: (r['parameters'], r['abs_rel']['mean']))
    report = {'summary': summary, 'B0': baseline, 'paired_SS1_minus_F3': pairs,
              'trajectories': curves, 'selected': selected['method'], 'rule': study['selection'],
              'eligible_within_relative_tie': [r['method'] for r in tied],
              'limits': ['One complete candidate; different inputs/capacity, not fusion-only attribution.',
                         'Validation reused for selection; exploratory scene bootstrap, no multiplicity correction.',
                         'Three-seed mean errors are not ensemble predictions; scene CI excludes seed uncertainty.',
                         'Only 577 fitting and 218 validation; no new test/refit/sample-efficiency or LoRA.'],
              'source_sha256': {str(Path(__file__).relative_to(ROOT)).replace('\\','/'): sha(Path(__file__))},
              'record_sha256': {str((r['path']/'result.json').relative_to(ROOT)).replace('\\','/'): sha(r['path']/'result.json')
                                for runs in groups.values() for r in runs}}
    write_json(out/'review.json', report)
    with (out/'summary.csv').open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['method', 'parameters', *KEYS])
        writer.writeheader()
        for row in summary:
            writer.writerow({'method': row['method'], 'parameters': row['parameters'], **{k: row[k]['mean'] for k in KEYS}})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for ax, key in zip(axes, ('abs_rel', 'boundary_f1', 'boundary_band_abs_rel')):
        ax.errorbar(range(4), [r[key]['mean'] for r in summary], yerr=[r[key]['sd'] for r in summary], fmt='o', capsize=5)
        ax.axhline(baseline[key], color='gray', ls='--', label='UniDepth B0')
        ax.set_xticks(range(4), [r['method'] for r in summary]); ax.set_title(key+' (mean +/- seed SD)'); ax.grid(alpha=.2); ax.legend()
    fig.savefig(out/'comparison.png', dpi=160); fig.savefig(out/'comparison.pdf'); plt.close(fig)
    a, b = averages['F3'], averages['SS1']
    diffs = [float(y['abs_rel'])-float(x['abs_rel']) for x, y in zip(a, b)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    axes[0].boxplot([[r['abs_rel'] for r in a], [r['abs_rel'] for r in b]], tick_labels=['F3', 'SS1'], showfliers=True)
    axes[0].set_ylabel('Per-image AbsRel (mean of 3 seeds)')
    axes[1].hist(diffs, bins=24); axes[1].axvline(0, color='red'); axes[1].set_xlabel('SS1 - F3 paired AbsRel; negative is better')
    fig.savefig(out/'paired_distribution.png', dpi=160); plt.close(fig)
    freeze = {'at': now(), 'status': 'selected_for_future_study_only', 'method': selected['method'],
              'selection': study['selection'], 'validation_abs_rel': selected['abs_rel']['mean'],
              'design_sha256': sha(ROOT/study['output']/'design_frozen.json'), 'evidence_sha256': sha(out/'review.json'),
              'previous_F3_freeze_untouched': True, 'no_refit_duration_applied': True}
    path = ROOT/study['output']/'residual_selection.json'
    if path.exists():
        previous = json.loads(path.read_text()); assert previous['method'] == freeze['method'] and previous['evidence_sha256'] == freeze['evidence_sha256']
    else:
        write_json(path, freeze)
    return report


def visuals_and_benchmark(study, groups):
    import h5py
    out = ROOT/study['output']; destination = out/'analysis/visuals'; destination.mkdir(exist_ok=True)
    meta = json.loads((out/'features/metadata.json').read_text())
    positions = {index: n for n, index in enumerate(meta['ids'])}
    arrays = {k: np.load(out/'features'/(k+'.npy'), mmap_mode='r') for k in study['candidate']['channels']}
    models, configs, scores = {}, {}, {}
    for name in ('F3', 'P3', 'G3', 'SS1'):
        run = groups[name][0]
        saved = torch.load(run['path']/'best.pt', map_location='cpu', weights_only=False)
        cfg = saved['config']['candidate']; configs[name] = cfg
        model = StyleSketchResidual(cfg) if name == 'SS1' else make_model(4, cfg, arrays)
        model.load_state_dict(saved['model']); models[name] = model.cuda().eval()
        scores[name] = {int(r['source_index']): r for r in run['scores']}
    ids = list(scores['SS1'])
    ranked = sorted(ids, key=lambda i: float(scores['SS1'][i]['abs_rel'])-float(scores['F3'][i]['abs_rel']))
    choices = {f'fixed_{sample:04d}': sample-1 for sample in study['fixed_visual_sample_ids']}
    choices.update(largest_gain=ranked[0], largest_regression=ranked[-1])
    checks = []; bench = []
    with NYUv2Labeled(ROOT/study['data']) as source, h5py.File(ROOT/study['parent_output']/'cache/development/predictions.h5', 'r') as cache, torch.inference_mode():
        for reason, index in choices.items():
            assert index in scores['SS1']
            sample = source.load(index+1); base = cache[str(index)]['original'][:]
            inp = observable_inputs(sample.rgb, base, None, {'drop_rgb': False, 'extra_inputs': []}, study['feature_hw'], {})
            inputs = torch.from_numpy(inp[None]).cuda(); bases = torch.from_numpy(base[None, None]).cuda()
            predicted = {'B0': base}
            for name, model in models.items():
                names = study['candidate']['channels'] if name == 'SS1' else configs[name]['branches']
                latent = {k: torch.from_numpy(np.array(arrays[k][positions[index]:positions[index]+1], copy=True)).float().cuda() for k in names}
                with torch.autocast('cuda', dtype=torch.float16):
                    residual = model(inputs, latent)
                predicted[name] = corrected_depth(bases, residual, 'additive')[0, 0].cpu().numpy()
                del latent
            pack = scoring_pack(sample, index)
            measured = {k: score_image(v, pack, 0) for k, v in predicted.items()}
            diff = {k: abs(measured[k]['abs_rel']-float(scores[k][index]['abs_rel'])) for k in models}
            assert max(diff.values()) < 1e-4, (index, diff)
            fig, axes = plt.subplots(3, 4, figsize=(14, 10), constrained_layout=True)
            axes[0, 0].imshow(sample.rgb); axes[0, 0].set_title('RGB')
            for ax, values, name in zip(axes[0, 1:], (sample.depth_m, predicted['F3'], predicted['SS1']), ('GT', 'F3', 'SS1')):
                ax.imshow(values, vmin=0, vmax=10, cmap='viridis'); ax.set_title(name+' | 0-10m')
            axes[1, 0].imshow(pack['mask'][0], cmap='gray'); axes[1, 0].set_title('Course evaluation mask')
            axes[1, 1].text(.02, .5, '\n'.join(f'{k}: AbsRel={measured[k]["abs_rel"]:.5f}\nF1={measured[k]["boundary_f1"]:.5f}' for k in ('F3','SS1')), fontsize=11)
            for ax, name in zip(axes[1, 2:], ('F3', 'SS1')):
                value = predicted[name]
                error = np.where(pack['mask'][0], np.abs(np.clip(value, .001, 10)-sample.depth_m)/np.maximum(sample.depth_m, .001), np.nan)
                ax.imshow(error, vmin=0, vmax=.5, cmap='magma'); ax.set_title(name+' AbsRel | 0-0.5')
            axes[2, 0].imshow(predicted['SS1']-predicted['F3'], vmin=-1, vmax=1, cmap='coolwarm'); axes[2, 0].set_title('SS1-F3 | -1 to 1m')
            for ax, name in zip(axes[2, 1:], ('GT', 'F3', 'SS1')):
                value = sample.depth_m if name == 'GT' else np.clip(predicted[name], .001, 10)
                _, _, gt_edges, edges = boundary_masks(value, sample.depth_m, pack['mask'][0], pack['mask'][0])
                ax.imshow(gt_edges if name == 'GT' else edges, cmap='gray'); ax.set_title(name+' fixed-threshold depth edges')
            for ax in axes.flat:
                ax.axis('off')
            fig.suptitle(f'{reason} | nyuv2_{index+1:04d} | seed320112; no per-image normalization')
            path = destination/f'{reason}_nyuv2_{index+1:04d}.png'; fig.savefig(path, dpi=130); plt.close(fig)
            fig, axes = plt.subplots(1, 6, figsize=(18, 3.2), constrained_layout=True)
            for ax, name in zip(axes, ('GT', 'B0', 'F3', 'P3', 'G3', 'SS1')):
                ax.imshow(sample.depth_m if name == 'GT' else predicted[name], vmin=0, vmax=10, cmap='viridis'); ax.set_title(name); ax.axis('off')
            fig.savefig(destination/f'{reason}_all_methods.png', dpi=120); plt.close(fig)
            checks.append({'reason': reason, 'sample_id': index+1, 'metrics': measured, 'saved_score_difference': diff,
                           'figure': path.relative_to(ROOT).as_posix(), 'sha256': sha(path)})
        # Same hardware/session, resident-input batch8, exclude foundation and IO.
        index = study['fixed_visual_sample_ids'][0]-1
        sample = source.load(index+1); base = cache[str(index)]['original'][:]
        inp = observable_inputs(sample.rgb, base, None, {'drop_rgb': False, 'extra_inputs': []}, study['feature_hw'], {})
        inputs = torch.from_numpy(np.repeat(inp[None], 8, axis=0)).cuda()
        bases = torch.from_numpy(np.repeat(base[None, None], 8, axis=0)).cuda()
        del residual
        for name, model in models.items():
            names = study['candidate']['channels'] if name == 'SS1' else configs[name]['branches']
            latent = {k: torch.from_numpy(np.repeat(np.array(arrays[k][positions[index]:positions[index]+1], copy=True), 8, axis=0)).float().cuda() for k in names}
            torch.cuda.empty_cache(); torch.cuda.synchronize(); initial = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
            durations = []
            for iteration in range(40):
                torch.cuda.synchronize(); tick = time.perf_counter()
                with torch.autocast('cuda', dtype=torch.float16):
                    output = corrected_depth(bases, model(inputs, latent), 'additive')
                torch.cuda.synchronize()
                if iteration >= 10:
                    durations.append((time.perf_counter()-tick)/8)
                del output
            bench.append({'method': name, 'batch_size': 8, 'warmups': 10, 'measured_batches': 30,
                          'seconds_per_image_median': float(np.median(durations)), 'seconds_per_image_mean': float(np.mean(durations)),
                          'resident_allocated_mib': initial/2**20, 'peak_incremental_mib': (torch.cuda.max_memory_allocated()-initial)/2**20,
                          'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
                          'scope': 'Cached refiner forward+restore; repeated fixed image batch; all four small model weights resident. Excludes foundation, transfer, feature extraction and IO.'})
            del latent
    write_json(out/'analysis/visuals/metadata.json', checks)
    write_json(out/'analysis/benchmark.json', bench)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--visuals', action='store_true')
    args = parser.parse_args()
    study = load(ROOT); check_freeze(ROOT, study)
    # Audit without rewriting the pretraining reference audit frozen by hash.
    groups = collect(study)
    report = summarize(study, groups)
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    if args.visuals:
        visuals_and_benchmark(study, groups)
        end_to_end_benchmark(study, groups)
    print(json.dumps({'selected': report['selected'], 'summary': report['summary'], 'paired': report['paired_SS1_minus_F3']}), flush=True)


def end_to_end_benchmark(study, groups):
    """Common batch1 deployment boundary, including frozen foundation and transfers."""
    import gc
    import yaml
    from p3_depth.models.runtime import ImageModel
    from p3_depth.stylesketch_residual.data import Capture
    gc.collect(); torch.cuda.empty_cache()
    settings = yaml.safe_load((ROOT/'configs/paths.local.yaml').read_text())
    adapter = ImageModel('unidepth', ROOT, settings, 'cuda', precision='fp16', threads=4)
    adapter.model.eval().requires_grad_(False)
    capture = Capture(adapter.model)
    with NYUv2Labeled(ROOT/study['data']) as source:
        images = [source.load(i).rgb for i in study['fixed_visual_sample_ids']]
    rows = []
    for name in ('B0', 'F3', 'SS1'):
        model = None
        if name != 'B0':
            saved = torch.load(groups[name][0]['path']/'best.pt', map_location='cpu', weights_only=False)
            cfg = saved['config']['candidate']
            channels = {'init': np.empty((0, 256)), 'mid': np.empty((0, 128)), 'late': np.empty((0, 64))}
            model = (StyleSketchResidual(cfg) if name == 'SS1' else make_model(4, cfg, channels)).cuda().eval()
            del saved
        capture.values.clear(); gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        times = []
        with torch.inference_mode():
            for iteration in range(24):
                rgb = images[iteration % len(images)]
                torch.cuda.synchronize(); tick = time.perf_counter()
                result = adapter.model.infer(torch.from_numpy(rgb.copy()).permute(2, 0, 1))
                base = result['depth'][0, 0].float().cpu().numpy()
                if model is not None:
                    inputs = observable_inputs(rgb, base, None, {'drop_rgb': False, 'extra_inputs': []}, study['feature_hw'], {})
                    names = study['candidate']['channels'] if name == 'SS1' else cfg['branches']
                    latent = {k: (result['depth_features'] if k == 'init' else capture.values[k]).float() for k in names}
                    with torch.autocast('cuda', dtype=torch.float16):
                        residual = model(torch.from_numpy(inputs[None]).cuda(), latent)
                    predicted = corrected_depth(result['depth'].float(), residual, 'additive')[0, 0].cpu().numpy()
                    assert np.isfinite(predicted).all()
                    del latent, residual
                torch.cuda.synchronize()
                if iteration >= 6:
                    times.append(time.perf_counter()-tick)
                del result
        rows.append({'method': name, 'batch_size': 1, 'warmups': 6, 'measured_images': 18,
                     'seconds_per_image_mean': float(np.mean(times)), 'seconds_per_image_median': float(np.median(times)),
                     'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
                     'scope': 'Resident RGB CPU array through official UniDepth inference, optional CPU RGB/depth preprocessing, GPU refiner and CPU depth output; excludes disk/model loading. Hooks active for all methods.'})
        del model
    capture.close()
    write_json(ROOT/study['output']/'analysis/end_to_end_benchmark.json', rows)


if __name__ == '__main__':
    main()
