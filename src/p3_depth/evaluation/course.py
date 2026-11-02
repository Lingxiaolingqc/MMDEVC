"""Spring 2027 revised course protocol (PDF received 2026-11-02).

Keep the previous unclipped evaluation available for historical experiments.
This profile evaluates raw metric predictions, never relative/GT-aligned values.
"""

import numpy as np

from p3_depth.data.nyuv2 import evaluation_mask
from p3_depth.evaluation.depth import depth_values_metrics


PROTOCOL = {
    'id': 'spring2027_revision_20261031',
    'output_size_hw': [480, 640],
    'crop_yx_half_open': [45, 471, 41, 601],
    'gt_range_m_inclusive': [0.001, 10.0],
    'prediction_clip_m': [0.001, 10.0],
    'alignment': 'none',
    'aggregation': 'equal mean of all 654 per-image scores',
    'metrics': ['abs_rel', 'rmse_m', 'delta1'],
    'invalid_policy': 'fail on nonfinite predictions in the GT evaluation region; do not drop pixels/images',
}


def score_course_nyu(raw, gt):
    """Inputs must already be metric meters at original NYU image size."""
    raw, gt = np.asarray(raw), np.asarray(gt)
    if raw.shape != (480, 640) or gt.shape != raw.shape:
        raise ValueError('Restore predictions to 480x640 before course evaluation')
    if not np.issubdtype(raw.dtype, np.floating):
        raise ValueError('Use raw floating-point metric predictions')
    region = evaluation_mask(gt)
    if not region.any():
        raise ValueError('Empty GT region; never omit an image from the 654-image mean')
    selected = raw[region].astype(np.float64)
    if not np.isfinite(selected).all():
        raise ValueError('Nonfinite prediction in fixed GT region; cannot silently omit it')
    clipped = np.clip(selected, 0.001, 10.0)
    scores = depth_values_metrics(clipped, gt[region].astype(np.float64))
    return {
        'gt_pixels': int(region.sum()),
        'clipped_low_pixels': int((selected < 0.001).sum()),
        'clipped_high_pixels': int((selected > 10).sum()),
        **{key: scores[key] for key in PROTOCOL['metrics']},
    }


def summarize_course(records, expected_images=654):
    if len(records) != expected_images or len({r['sample_id'] for r in records}) != expected_images:
        raise ValueError('Every declared image must contribute exactly once')
    return {
        'images': expected_images,
        'gt_pixels': sum(r['gt_pixels'] for r in records),
        'clipped_low_pixels': sum(r['clipped_low_pixels'] for r in records),
        'clipped_high_pixels': sum(r['clipped_high_pixels'] for r in records),
        **{key: float(np.mean([r[key] for r in records])) for key in PROTOCOL['metrics']},
    }
