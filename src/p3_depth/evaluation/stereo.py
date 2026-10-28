"""Single-pair diagnostics on fixed GT regions, with explicit invalid penalties."""

import numpy as np

from p3_depth.evaluation.depth import depth_values_metrics


def evaluate_region(prediction_depth: np.ndarray, prediction_disparity: np.ndarray,
                    prediction_valid: np.ndarray, gt_depth: np.ndarray,
                    gt_disparity: np.ndarray, region: np.ndarray) -> dict:
    """Depth errors condition on valid predictions; bad disparity counts invalids as bad.

    The caller must define region from GT alone. No crop, clipping or scale alignment.
    An empty GT region is an error; zero prediction coverage yields null depth errors.
    """
    arrays = (prediction_depth, prediction_disparity, prediction_valid,
              gt_depth, gt_disparity, region)
    if region.ndim != 2 or any(a.shape != region.shape for a in arrays):
        raise ValueError("All arrays must have the same H x W shape")
    if region.dtype != np.bool_ or prediction_valid.dtype != np.bool_:
        raise ValueError("Region and prediction masks must be boolean")
    physical = (np.isfinite(prediction_depth) & (prediction_depth > 0)
                & np.isfinite(prediction_disparity))
    if np.any(prediction_valid & ~physical):
        raise ValueError("Prediction mask includes nonfinite or nonpositive values")
    if np.any(region & (~np.isfinite(gt_depth) | (gt_depth <= 0) | ~np.isfinite(gt_disparity))):
        raise ValueError("GT region includes invalid depth/disparity")
    count = int(region.sum())
    if count == 0:
        raise ValueError("Cannot evaluate an empty GT region")
    selected = region & prediction_valid
    predicted_count = int(selected.sum())
    invalid_count = count - predicted_count
    disparity_error = np.abs(prediction_disparity[selected].astype(np.float64)
                             - gt_disparity[selected].astype(np.float64))
    result = {
        "gt_pixels": count, "valid_prediction_pixels": predicted_count,
        "invalid_prediction_pixels": invalid_count,
        "coverage": predicted_count / count, "invalid_fraction": invalid_count / count,
        "disparity_mae_px_on_valid": float(disparity_error.mean()) if predicted_count else None,
        "bad1_fraction_invalid_as_bad": (int((disparity_error > 1).sum()) + invalid_count) / count,
        "bad2_fraction_invalid_as_bad": (int((disparity_error > 2).sum()) + invalid_count) / count,
    }
    result["depth_on_valid_predictions"] = depth_values_metrics(prediction_depth[selected], gt_depth[selected])
    return result
