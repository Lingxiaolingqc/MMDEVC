"""Single-pair diagnostics on fixed GT regions, with explicit invalid penalties."""

import numpy as np

from p3_depth.evaluation.depth import depth_values_metrics


def compare_lr_rejection(before, after, before_valid, after_valid, gt, region):
    """Explain the coverage trade-off of filtering; never treat discarded values as corrected."""
    if any(x.shape != before.shape for x in (after,before_valid,after_valid,gt,region)):
        raise ValueError('LR arrays must share the same image grid')
    if any(x.dtype != np.bool_ for x in (before_valid,after_valid,region)):
        raise ValueError('LR masks must be boolean')
    if np.any(after_valid & ~before_valid) or not np.array_equal(before[after_valid],after[after_valid]):
        raise ValueError('LR filtering must retain a subset without changing its values')
    if np.any(region & ~np.isfinite(gt)) or np.any(before_valid & ~np.isfinite(before)):
        raise ValueError('Valid GT/predictions must be finite')
    removed=region & before_valid & ~after_valid
    errors=np.abs(before[removed].astype(np.float64)-gt[removed])
    n=int(removed.sum());bad=int((errors>1).sum())
    return {'removed_gt_pixels':n,'removed_bad_gt1px_pixels':bad,'removed_within_1px_pixels':n-bad,
            'removed_bad_gt1px_fraction':bad/n if n else None,
            'removed_disparity_mae_px':float(errors.mean()) if n else None,
            'retained_values_unchanged':True}


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
