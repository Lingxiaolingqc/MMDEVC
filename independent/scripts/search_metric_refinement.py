"""Compare fixed-resolution metric refinement on scene-held-out NYU development data."""
import json
from pathlib import Path
import time

import h5py
import numpy as np

from depth_project.metrics import average_image_metrics, evaluate_nyu
from depth_project.refinement import blend_flip, fit_global_scale, guided_filter


def calibration_support(prediction, truth):
    prediction = prediction[45:471:8, 41:601:8]
    truth = truth[45:471:8, 41:601:8]
    valid = np.isfinite(truth) & (truth > .001) & (truth < 10)
    return np.maximum(prediction[valid], .001), truth[valid]


def main():
    root = Path(__file__).resolve().parents[1]
    cache = root / 'runs/nyu-development'
    manifest = json.loads((cache / 'manifest.json').read_text())
    assert manifest.get('completed_predictions') == 795, 'Complete the development cache first'
    output = root / 'runs/metric-search'
    output.mkdir(parents=True, exist_ok=True)
    weights = [0, .25, .5, .75, 1.]
    scales = {}
    with h5py.File(root / 'data/nyu/nyu_depth_v2_labeled.mat') as dataset:
        fitting_support = []
        for index in manifest['fit_ids']:
            with np.load(cache / f'prediction_{index:04d}.npz') as sample:
                truth = dataset['depths'][index].T
                original, valid_truth = calibration_support(sample['depth_m'], truth)
                flip, _ = calibration_support(sample['flip_m'], truth)
                fitting_support.append((original, flip, valid_truth))
        for weight in weights:
            predictions = []
            truths = []
            for original, flip, truth in fitting_support:
                predictions.append(blend_flip(original, flip, weight))
                truths.append(truth)
            scales[str(weight)] = fit_global_scale(predictions, truths)
            print(f'Fit flip weight {weight}: global scale {scales[str(weight)]:.6f}', flush=True)
        candidates = []
        for weight in weights:
            for scale in [1., scales[str(weight)]]:
                candidates.append({'flip_weight': weight, 'scale': scale, 'radius': 0, 'epsilon': .001})
        for weight in [0, .5]:
            for radius in [2, 4, 8]:
                for epsilon in [.0001, .001, .01]:
                    candidates.append({'flip_weight': weight, 'scale': scales[str(weight)], 'radius': radius, 'epsilon': epsilon})
        rows = []
        samples = []
        for index in manifest['validation_ids']:
            with np.load(cache / f'prediction_{index:04d}.npz') as sample:
                original = sample['depth_m'].copy()
                flip = sample['flip_m'].copy()
            samples.append((index, original, flip, np.ascontiguousarray(dataset['images'][index].transpose(2, 1, 0)), dataset['depths'][index].T))
        for config in candidates:
            per_image = []
            start = time.perf_counter()
            for index, original, flip, rgb, truth in samples:
                prediction = blend_flip(original, flip, config['flip_weight']) * config['scale']
                if config['radius']:
                    prediction = guided_filter(rgb, prediction, config['radius'], config['epsilon'])
                per_image.append({'image_id': index, **evaluate_nyu(prediction, truth)})
            row = {'config': config, **average_image_metrics(per_image), 'replay_s': time.perf_counter() - start}
            rows.append(row)
            print(json.dumps(row), flush=True)
        winner = min(rows, key=lambda row: row['abs_rel'])
        record = {'split': {'fit_ids': manifest['fit_ids'], 'validation_ids': manifest['validation_ids'], 'validation_scenes': manifest['validation_scenes']}, 'identities': manifest['identities'], 'selection': 'lowest validation AbsRel, scene-disjoint development only', 'rows': rows, 'best_simple': winner, 'global_scales': scales}
        (output / 'simple-search.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        (root / 'results/metric-development-search.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        print('BEST SIMPLE', json.dumps(winner), flush=True)


if __name__ == '__main__':
    main()
