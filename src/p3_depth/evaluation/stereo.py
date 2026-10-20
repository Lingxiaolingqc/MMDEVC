"""Single-pair diagnostics on fixed GT regions, with explicit invalid penalties."""

import numpy as np


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
    names = ["abs_rel", "sq_rel_m", "rmse_m", "rmse_log", "delta1", "delta2", "delta3"]
    if predicted_count == 0:
        result["depth_on_valid_predictions"] = dict.fromkeys(names)
        return result
    pred = prediction_depth[selected].astype(np.float64)
    gt = gt_depth[selected].astype(np.float64)
    error = pred - gt
    ratio = np.maximum(pred / gt, gt / pred)
    result["depth_on_valid_predictions"] = {
        "abs_rel": float(np.mean(np.abs(error) / gt)),
        "sq_rel_m": float(np.mean(error ** 2 / gt)),
        "rmse_m": float(np.sqrt(np.mean(error ** 2))),
        "rmse_log": float(np.sqrt(np.mean((np.log(pred) - np.log(gt)) ** 2))),
        **{f"delta{i}": float(np.mean(ratio < 1.25 ** i)) for i in (1, 2, 3)},
    }
    return result
