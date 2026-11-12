"""Frozen NYU improvement evaluation with paired baseline and ablations."""
import csv
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch
from safetensors.torch import load_file

from .data import load_nyu_split
from .metrics import average_image_metrics, evaluate_nyu
from .models import load_image_adapter
from .records import run_provenance, sha256_file
from .refinement import blend_flip, refine_metric
from .residual_refiner import ResidualRefiner, make_features, apply_residual


def verify_frozen_dataset(root, config):
    root = Path(root)
    paths = {'dataset_sha256': root / 'data/nyu/nyu_depth_v2_labeled.mat', 'split_sha256': root / 'data/nyu/splits.mat'}
    for name, path in paths.items():
        if sha256_file(path) != config[name]:
            raise ValueError(f'Frozen {name} identity differs from the evaluation input')


def historical_baseline_difference(path, prediction):
    path = Path(path)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as source:
        previous = source['depth_m']
        if previous.shape != prediction.shape or not np.isfinite(previous).all():
            raise ValueError('Historical baseline prediction has incompatible values')
        return float(np.max(np.abs(previous - prediction)))


def run_metric_evaluation(root, config_path, output):
    root, output = Path(root), Path(output)
    config = json.loads(Path(config_path).read_text())
    if config['selection_status'] != 'frozen on scene-held-out development before final test':
        raise ValueError('Freeze development selection before test evaluation')
    if (output / 'summary.json').exists():
        raise FileExistsError('Final metric evaluation already exists')
    verify_frozen_dataset(root, config)
    output.mkdir(parents=True, exist_ok=True)
    config_hash = sha256_file(config_path)
    checkpoint = root / config['refiner_checkpoint']
    if sha256_file(checkpoint) != config['refiner_sha256']:
        raise ValueError('Refiner checkpoint identity differs from frozen selection')
    provenance = run_provenance(root)
    torch.set_num_threads(4)
    adapter = load_image_adapter('unidepth', root)
    if adapter.metadata['checkpoint_sha256'] != config['foundation_checkpoint_sha256']:
        raise ValueError('Foundation checkpoint identity differs')
    refiner = ResidualRefiner().cuda().eval()
    refiner.load_state_dict(load_file(str(checkpoint), device='cuda'))
    split = load_nyu_split(root / 'data/nyu/splits.mat')
    assert len(split['test']) == 654
    assert set(config['fit_ids']).isdisjoint(config['validation_ids'])
    assert set(config['fit_ids'] + config['validation_ids']) == set(split['development'].tolist())
    assert set(config['fit_ids'] + config['validation_ids']).isdisjoint(split['test'].tolist())
    rows = {name: [] for name in ['baseline', 'global_scale', 'flip_only', 'selected_simple', 'residual_refiner']}
    latencies = []
    baseline_latencies = []
    maximum_baseline_discrepancy = 0.
    historical_comparison_count = 0
    scenes = []
    with h5py.File(root / 'data/nyu/nyu_depth_v2_labeled.mat') as dataset:
        warm = dataset['images'][int(split['development'][0])].transpose(2, 1, 0).copy()
        for _ in range(2):
            adapter.predict_rgb(warm)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for number, index in enumerate(split['test']):
            index = int(index)
            rgb = np.ascontiguousarray(dataset['images'][index].transpose(2, 1, 0))
            truth = dataset['depths'][index].T
            scene = ''.join(chr(int(value)) for value in dataset[dataset['scenes'][0, index]][:].reshape(-1))
            scenes.append(scene)
            tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).cuda()
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                original_outputs = adapter.model.infer(tensor)
                original = original_outputs['depth'][0, 0].float().cpu().numpy()
                confidence = original_outputs['confidence'][0, 0].float().cpu().numpy()
                torch.cuda.synchronize()
                baseline_latencies.append(time.perf_counter() - start)
                flipped_outputs = adapter.model.infer(torch.flip(tensor, [-1]))
                flip = torch.flip(flipped_outputs['depth'], [-1])[0, 0].float().cpu().numpy()
                simple = refine_metric(rgb, original, flip, config['simple_config'])
                features = make_features(rgb, original, flip, confidence, simple, tuple(config['feature_hw']))
                with torch.autocast('cuda', dtype=torch.float16):
                    residual = refiner(torch.from_numpy(features[None]).cuda())[0, 0].float().cpu().numpy()
                improved = apply_residual(simple, residual)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - start)
            historical_difference = historical_baseline_difference(root / f'runs/nyu/prediction_{index:04d}.npz', original)
            if historical_difference is not None:
                maximum_baseline_discrepancy = max(maximum_baseline_discrepancy, historical_difference)
                historical_comparison_count += 1
            predictions = {
                'baseline': original,
                'global_scale': original * config['original_global_scale'],
                'flip_only': blend_flip(original, flip, config['simple_config']['flip_weight']),
                'selected_simple': simple,
                'residual_refiner': improved,
            }
            for name, prediction in predictions.items():
                rows[name].append({'image_id': index, 'scene': scene, **evaluate_nyu(prediction, truth)})
            np.savez_compressed(output / f'prediction_{index:04d}.npz', depth_m=improved, baseline_m=original, simple_m=simple, image_id=index, configuration_sha256=config_hash)
            if (number + 1) % 50 == 0 or number == 653:
                print(f'Metric final {number+1}/654: base={np.mean([row["abs_rel"] for row in rows["baseline"]]):.6f} improved={np.mean([row["abs_rel"] for row in rows["residual_refiner"]]):.6f}', flush=True)
    metrics = {name: average_image_metrics(values) for name, values in rows.items()}
    unique_scenes = sorted(set(scenes))
    base_sums = np.array([sum(row['abs_rel'] for row in rows['baseline'] if row['scene'] == scene) for scene in unique_scenes])
    final_sums = np.array([sum(row['abs_rel'] for row in rows['residual_refiner'] if row['scene'] == scene) for scene in unique_scenes])
    rng = np.random.default_rng(320112)
    sampled = rng.integers(0, len(unique_scenes), (5000, len(unique_scenes)))
    gains = 100 * (base_sums[sampled].sum(axis=1) - final_sums[sampled].sum(axis=1)) / base_sums[sampled].sum(axis=1)
    delta = np.array([a['abs_rel'] - b['abs_rel'] for a, b in zip(rows['baseline'], rows['residual_refiner'])])
    summary = {
        'metrics': metrics, 'config': config, 'configuration_sha256': config_hash,
        'image_count': 654, 'test_ids': split['test'].tolist(),
        'abs_rel_reduction_percent': 100 * (1 - metrics['residual_refiner']['abs_rel'] / metrics['baseline']['abs_rel']),
        'scene_cluster_bootstrap_95_percent_reduction_interval': np.percentile(gains, [2.5, 97.5]).tolist(),
        'per_image_improved_count': int((delta > 0).sum()), 'per_image_worsened_count': int((delta < 0).sum()),
        'maximum_reproduced_baseline_pixel_difference_m': maximum_baseline_discrepancy if historical_comparison_count else None,
        'historical_baseline_comparison_count': historical_comparison_count,
        'timing': {'baseline_median_s': float(np.median(baseline_latencies)), 'improved_median_s': float(np.median(latencies)), 'peak_allocated_mib': torch.cuda.max_memory_allocated()/1024**2, 'peak_reserved_mib': torch.cuda.max_memory_reserved()/1024**2, 'scope': 'synchronized original+flip inference, feature preparation, refinement and residual resize; excludes file I/O and evaluation; two original-model warmups'},
        'hardware': torch.cuda.get_device_name(), 'provenance': provenance,
        'protocol': json.loads((root / 'configs/nyu-course.json').read_text()),
        'setting': 'NYU supervised development adaptation with scene-held-out selection; frozen UniDepthV2-S foundation weights',
    }
    for name, values in rows.items():
        with (output / f'{name}-per-image.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (root / 'results/metric-improvement.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2), flush=True)
    return summary
