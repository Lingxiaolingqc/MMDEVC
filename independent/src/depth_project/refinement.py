"""Development-fitted metric calibration and fixed-resolution refinement."""
import cv2
import numpy as np


def fit_global_scale(predictions, truths):
    """Minimize unclipped AbsRel with equal total weight per input image."""
    if len(predictions) != len(truths) or not len(predictions):
        raise ValueError('Calibration requires matching nonempty image lists')
    ratios = []
    weights = []
    for prediction, truth in zip(predictions, truths):
        prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
        truth = np.asarray(truth, dtype=np.float64).reshape(-1)
        if prediction.shape != truth.shape or not prediction.size:
            raise ValueError('Calibration support must match')
        if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
            raise ValueError('Calibration values must be finite')
        if (prediction <= 0).any() or (truth <= 0).any():
            raise ValueError('Calibration values must be positive')
        ratios.append(truth / prediction)
        weights.append(prediction / truth / len(prediction))
    ratios = np.concatenate(ratios)
    weights = np.concatenate(weights)
    order = np.argsort(ratios)
    cumulative = np.cumsum(weights[order])
    index = np.searchsorted(cumulative, cumulative[-1] / 2)
    return float(ratios[order[index]])


def blend_flip(original, restored_flip, flip_weight):
    original = np.asarray(original, dtype=np.float32)
    restored_flip = np.asarray(restored_flip, dtype=np.float32)
    if original.shape != restored_flip.shape or not 0 <= flip_weight <= 1:
        raise ValueError('Flip blending requires matching shapes and a weight in [0,1]')
    if not np.isfinite(original).all() or not np.isfinite(restored_flip).all():
        raise ValueError('Flip predictions must be finite')
    return (1 - flip_weight) * original + flip_weight * restored_flip


def guided_filter(rgb, depth, radius, epsilon):
    """Scalar guided filter using RGB luminance, in metric-depth units."""
    depth = np.asarray(depth, dtype=np.float32)
    if rgb.shape != (*depth.shape, 3) or radius < 1 or epsilon <= 0:
        raise ValueError('Guided filtering requires matching RGB/depth and positive settings')
    if not np.isfinite(depth).all():
        raise ValueError('Depth must be finite')
    guide = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    kernel = (2 * radius + 1, 2 * radius + 1)

    def mean(values):
        return cv2.boxFilter(values, -1, kernel, borderType=cv2.BORDER_REFLECT)

    mean_guide = mean(guide)
    mean_depth = mean(depth)
    covariance = mean(guide * depth) - mean_guide * mean_depth
    variance = np.maximum(mean(guide * guide) - mean_guide ** 2, 0)
    slope = covariance / (variance + epsilon)
    offset = mean_depth - slope * mean_guide
    return mean(slope) * guide + mean(offset)


def refine_metric(rgb, original, flip, config):
    scale = float(config['scale'])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Metric scale must be finite and positive')
    result = blend_flip(original, flip, config['flip_weight']) * scale
    if config.get('radius', 0):
        result = guided_filter(rgb, result, config['radius'], config['epsilon'])
    return np.maximum(result, .001)
