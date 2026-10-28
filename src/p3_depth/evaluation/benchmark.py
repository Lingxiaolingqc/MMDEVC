"""Fixed NYUv2 sample selection and explicit image-wise metric aggregation."""

import csv
import numpy as np

from p3_depth.data.nyuv2 import evaluation_mask
from p3_depth.evaluation.depth import prepare_prediction, evaluate_depth


def select_nyu(root, subset, limit=None):
    path = root / ('manifests/nyuv2_dev.csv' if subset == 'dev' else 'manifests/nyuv2.csv')
    with path.open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r['split'] == ('train' if subset == 'dev' else 'test')]
    if len(rows) != (3 if subset == 'dev' else 654) or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Expected exactly 3 train development IDs or 654 unique standard test IDs')
    if limit is not None:
        if limit < 1 or limit > len(rows):
            raise ValueError('Limit must select between 1 and the full subset size')
        rows = rows[:limit]
    return path, rows


def score_raw(raw, gt, valid, asset):
    region = evaluation_mask(gt)
    depth, mask, transform = prepare_prediction(raw, gt, region, asset['prediction_type'], asset['unit'], asset['alignment'], valid)
    return evaluate_depth(depth, gt, region, mask), transform


def summarize(records):
    if not records:
        raise ValueError('Cannot summarize an empty benchmark')
    scores = [r['metrics'] for r in records]
    names = list(scores[0]['depth_on_valid_predictions'])
    means = {}
    counts = {}
    for key in names:
        values = [s['depth_on_valid_predictions'][key] for s in scores if s['depth_on_valid_predictions'][key] is not None]
        means[key] = float(np.mean(values)) if values else None
        counts[key] = len(values)
    total = sum(s['gt_pixels'] for s in scores)
    predicted = sum(s['valid_prediction_pixels'] for s in scores)
    times = [r['inference_seconds'] for r in records]
    return {'images': len(records), 'image_macro_metrics_on_valid': means, 'contributing_images': counts,
            'zero_coverage_images': sum(s['valid_prediction_pixels'] == 0 for s in scores),
            'pixel_pooled_coverage': predicted / total, 'gt_pixels': total, 'valid_prediction_pixels': predicted,
            'image_macro_full_gt_delta': {k: float(np.mean([s['delta_full_gt_invalid_as_failure'][k] for s in scores])) for k in ('delta1','delta2','delta3')},
            'mean_infer_seconds': float(np.mean(times)), 'throughput_fps_excluding_io': len(times) / sum(times),
            'aggregation': 'unweighted mean of per-image metrics; undefined errors excluded with counts, never converted to zero'}
