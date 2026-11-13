"""Regularized image-specific calibration from frozen visual features."""
import numpy as np


def global_features(depth_features, base):
    depth_features = np.asarray(depth_features, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    if depth_features.ndim != 3 or base.ndim != 2:
        raise ValueError("Calibration features require CHW latents and HW depth")
    if not np.isfinite(depth_features).all() or not np.isfinite(base).all() or (base <= 0).any():
        raise ValueError("Calibration inputs must be finite with positive depth")
    log_depth = np.log(base)
    pooled = depth_features.mean(axis=(1, 2))
    statistics = np.r_[np.quantile(log_depth, [0.1, 0.25, 0.5, 0.75, 0.9]), log_depth.mean(), log_depth.std()]
    return np.r_[pooled, statistics]


def fit_ridge_scale(features, scales, alpha):
    features = np.asarray(features, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    if features.ndim != 2 or scales.shape != (len(features),) or not len(features) or alpha <= 0:
        raise ValueError("Ridge calibration requires matching nonempty arrays and positive regularization")
    if not np.isfinite(features).all() or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Ridge fitting values must be finite with positive scales")
    mean = features.mean(axis=0)
    standard_deviation = features.std(axis=0)
    standard_deviation = np.where(standard_deviation > 1e-6, standard_deviation, 1)
    design = (features - mean) / standard_deviation
    target = np.log(scales)
    target_mean = float(target.mean())
    weights = np.linalg.solve(design.T @ design + alpha * np.eye(design.shape[1]), design.T @ (target - target_mean))
    return {"mean": mean, "standard_deviation": standard_deviation, "weights": weights,
            "target_mean": target_mean, "alpha": alpha, "fitting_count": len(features), "log_scale_bounds": [-0.5, 0.5]}


def predict_ridge_scale(model, features):
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != len(model["mean"]) or not np.isfinite(features).all():
        raise ValueError("Calibration inference feature dimensions or values differ")
    normalized = (features - np.asarray(model["mean"])) / np.asarray(model["standard_deviation"])
    log_scale = normalized @ np.asarray(model["weights"]) + model["target_mean"]
    return np.exp(np.clip(log_scale, *model["log_scale_bounds"]))
