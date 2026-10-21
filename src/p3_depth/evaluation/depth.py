"""Depth metrics and explicit benchmark-only relative alignment, in NumPy."""

import numpy as np


def depth_values_metrics(pred, gt):
    """Positive finite, already-selected 1D depths; natural log, strict delta thresholds."""
    names = ["abs_rel", "sq_rel_m", "rmse_m", "rmse_log", "delta1", "delta2", "delta3"]
    pred, gt = np.asarray(pred, dtype=np.float64), np.asarray(gt, dtype=np.float64)
    if pred.ndim != 1 or pred.shape != gt.shape:
        raise ValueError("Selected depth vectors must have identical 1D shape")
    if not (np.isfinite(pred).all() and np.isfinite(gt).all()) or np.any(pred <= 0) or np.any(gt <= 0):
        raise ValueError("Metric vectors must be positive and finite")
    if len(pred) == 0:
        return dict.fromkeys(names)
    error = pred - gt
    ratio = np.maximum(pred / gt, gt / pred)
    return {"abs_rel": float(np.mean(np.abs(error) / gt)),
            "sq_rel_m": float(np.mean(error ** 2 / gt)),
            "rmse_m": float(np.sqrt(np.mean(error ** 2))),
            "rmse_log": float(np.sqrt(np.mean((np.log(pred) - np.log(gt)) ** 2))),
            **{f"delta{i}": float(np.mean(ratio < 1.25 ** i)) for i in (1, 2, 3)}}


def prepare_prediction(raw, gt, region, prediction_type, unit, alignment, valid=None):
    """No implicit resizing/clipping. Alignment is per-image on GT for evaluation only.

    metric_depth + none: m/mm conversion only.
    relative_depth + median: positive depth-like values scaled by median GT / median prediction.
    inverse_depth_like + inverse_scale_shift: fit a*raw+b to 1/GT; invert positive results.
    """
    raw, gt, region = np.asarray(raw), np.asarray(gt), np.asarray(region)
    if not np.issubdtype(raw.dtype, np.floating):
        raise ValueError("Use raw floating-point predictions, not integer/colorized depth")
    if raw.ndim != 2 or raw.shape != gt.shape or raw.shape != region.shape or region.dtype != np.bool_:
        raise ValueError("Prediction, GT and boolean region must share original H x W shape; resize explicitly upstream")
    if np.any(region & (~np.isfinite(gt) | (gt <= 0))) or not region.any():
        raise ValueError("GT region must be nonempty, finite and positive")
    if valid is None:
        valid = np.isfinite(raw)
    if valid.shape != raw.shape or valid.dtype != np.bool_:
        raise ValueError("Prediction mask must be boolean and match prediction shape")
    valid = valid & np.isfinite(raw)
    pred = raw.astype(np.float64, copy=True)
    parameters = {"prediction_type": prediction_type, "input_unit": unit, "alignment": alignment,
                  "gt_used_for_alignment": alignment != "none"}
    if prediction_type == "metric_depth":
        if alignment != "none" or unit not in {"m", "mm"}:
            raise ValueError("Direct metric requires alignment=none and unit=m/mm")
        pred *= 0.001 if unit == "mm" else 1.0
        parameters["unit_scale"] = 0.001 if unit == "mm" else 1.0
        label = "direct_metric"
    elif prediction_type == "relative_depth":
        if unit != "arbitrary" or alignment != "median":
            raise ValueError("Depth-like relative output requires arbitrary units and explicit median alignment")
        valid &= pred > 0
        fit = region & valid
        if not fit.any():
            raise ValueError("No positive relative predictions available for alignment")
        scale = float(np.median(gt[fit]) / np.median(pred[fit]))
        pred *= scale
        parameters.update(scale=scale, shift=0., fit_pixels=int(fit.sum()), fit_domain="depth")
        label = "relative_aligned"
    elif prediction_type == "inverse_depth_like":
        if unit != "arbitrary" or alignment != "inverse_scale_shift":
            raise ValueError("Inverse-like output requires arbitrary units and explicit inverse_scale_shift")
        fit = region & valid
        if fit.sum() < 2 or np.std(pred[fit]) < 1e-12:
            raise ValueError("Degenerate inverse-depth alignment: insufficient variation")
        matrix = np.column_stack((pred[fit], np.ones(int(fit.sum()))))
        coefficients, _, rank, _ = np.linalg.lstsq(matrix, 1.0 / gt[fit], rcond=None)
        scale, shift = map(float, coefficients)
        if rank != 2 or not np.isfinite(coefficients).all() or scale <= 0:
            raise ValueError("Inverse-depth fit has wrong direction or is degenerate")
        inverse = scale * pred + shift
        valid &= np.isfinite(inverse) & (inverse > 0)
        pred = np.divide(1., inverse, out=np.full_like(inverse, np.nan), where=valid)
        parameters.update(scale=scale, shift=shift, fit_pixels=int(fit.sum()), fit_domain="inverse_depth")
        label = "relative_aligned"
    else:
        raise ValueError("Unknown prediction representation")
    with np.errstate(over="ignore", invalid="ignore"):
        pred = pred.astype(np.float32)
    valid &= np.isfinite(pred) & (pred > 0)
    pred[~valid] = np.nan
    parameters["result_label"] = label
    return pred, valid, parameters


def evaluate_depth(pred, gt, region, valid=None):
    if pred.ndim != 2 or pred.shape != gt.shape or pred.shape != region.shape or region.dtype != np.bool_:
        raise ValueError("Prediction, GT and boolean region must share H x W shape")
    if not region.any() or np.any(region & (~np.isfinite(gt) | (gt <= 0))):
        raise ValueError("GT region must be nonempty, finite and positive")
    if valid is None:
        valid = np.isfinite(pred) & (pred > 0)
    if valid.dtype != np.bool_ or valid.shape != pred.shape:
        raise ValueError("Invalid prediction mask")
    if np.any(valid & (~np.isfinite(pred) | (pred <= 0))):
        raise ValueError("Prediction mask includes invalid values")
    chosen = region & valid
    count, predicted = int(region.sum()), int(chosen.sum())
    scores = depth_values_metrics(pred[chosen], gt[chosen])
    return {"gt_pixels": count, "valid_prediction_pixels": predicted,
            "invalid_prediction_pixels": count - predicted, "coverage": predicted/count,
            "invalid_fraction": 1-predicted/count, "depth_on_valid_predictions": scores,
            "delta_full_gt_invalid_as_failure": {f"delta{i}": (scores[f"delta{i}"] * predicted/count if predicted else 0.)
                                                   for i in (1, 2, 3)}}
