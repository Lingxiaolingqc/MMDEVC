"""Bounded E4 depth-penalty refinement; original E0-E4 implementation unchanged."""
import cv2
import numpy as np
from p3_depth.temporal.fusion import edge_strength, fuse_step, fuse_sequence
from p3_depth.temporal.flow import sample_previous


def fixed_regions(current, fields, parameters, setting):
    """Current DA/flow-only masks, identical between original/global/selective modes."""
    valid = np.isfinite(current)
    finite = current[valid]
    span = float(np.diff(np.quantile(finite, [.01, .99]))[0]) if len(finite) else 0.
    edge = edge_strength(current, valid) / max(span, 1e-8)
    radius = int(setting["edge_guard_radius_px"])
    if radius < 0:
        raise ValueError("Negative guard radius")
    band = edge > setting["edge_guard_normalized"]
    band = cv2.dilate(band.astype(np.uint8), np.ones((2*radius+1, 2*radius+1), np.uint8)) > 0
    sampled_forward, inside = sample_previous(fields["forward"], fields["backward"])
    norm_squared = np.sum(fields["backward"]**2 + sampled_forward**2, axis=2)
    hard = (valid & inside & np.isfinite(fields["photo_residual"])
            & (fields["photo_residual"] <= parameters["photo_max"])
            & (fields["fb_residual"]**2 <= parameters["fb_relative"]*norm_squared + parameters["fb_absolute"]))
    reliable = (hard & ~band & (fields["photo_residual"] <= setting["reliable_photo_max"])
                & (fields["fb_residual"] <= setting["reliable_fb_max_px"]) & (span > 1e-8))
    return {"valid": valid, "hard": hard, "edge_band": band,
            "reliable": reliable, "edge": edge, "span": span}


def refinement_step(current, history, parameters, fields, setting, mode):
    if mode not in ("original", "global", "selective"):
        raise ValueError("Unknown refinement mode")
    factor = float(setting["depth_sigma_factor"])
    if not np.isfinite(factor) or factor < 1:
        raise ValueError("Relaxation factor must be finite and >=1")
    args = {k: fields[k] for k in ("backward", "forward", "photo_residual", "fb_residual")}
    if mode == "original" or factor == 1:
        return fuse_step(current, history, "E4", parameters,
                         representation="inverse_depth_like", **args)
    relaxed = {**parameters, "depth_span_sigma": parameters["depth_span_sigma"] * factor}
    broad, broad_weight = fuse_step(current, history, "E4", relaxed,
                                    representation="inverse_depth_like", **args)
    if mode == "global":
        return broad, broad_weight
    original, old_weight = fuse_step(current, history, "E4", parameters,
                                     representation="inverse_depth_like", **args)
    mask = fixed_regions(current, fields, parameters, setting)["reliable"]
    return (np.where(mask, broad, original).astype(np.float32),
            np.where(mask, broad_weight, old_weight).astype(np.float32))


def refinement_sequence(depths, valid, parameters, flows, setting, mode):
    if depths.ndim != 3 or valid.shape != depths.shape or valid.dtype != np.bool_:
        raise ValueError("Expected THW depths and boolean validity")
    if mode == "original":
        return fuse_sequence(depths, valid, "E4", parameters, flows, representation="inverse_depth_like")
    raw = np.where(valid & np.isfinite(depths), depths, np.nan).astype(np.float32)
    result, weights = raw.copy(), np.zeros_like(raw)
    for t in range(1, len(raw)):
        fields = {k: flows[k][t-1] for k in ("backward", "forward", "photo_residual", "fb_residual")}
        result[t], weights[t] = refinement_step(raw[t], result[t-1], parameters, fields, setting, mode)
    return result, weights

