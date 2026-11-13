"""Scene-separated research splits, training diagnostics and metric losses."""
import cv2
import numpy as np
import torch

from .refinement import fit_global_scale


def nested_scene_subsets(scenes, targets, seed):
    if not scenes or any(target < 1 or target > len(scenes) for target in targets):
        raise ValueError("Subset targets must fit the available image pool")
    names = np.array(sorted(set(scenes.values())))
    np.random.default_rng(seed).shuffle(names)
    ordered = [index for name in names for index in sorted(scenes) if scenes[index] == name]
    result = {}
    for target in targets:
        end = target
        while end < len(ordered) and scenes[ordered[end]] == scenes[ordered[end - 1]]:
            end += 1
        result[target] = sorted(ordered[:end])
    return result


def scene_folds(scenes, count, seed):
    names = np.array(sorted(set(scenes.values())))
    if count < 2 or len(names) < count:
        raise ValueError("Folds require at least two nonempty scene groups")
    np.random.default_rng(seed).shuffle(names)
    return [sorted(index for index, scene in scenes.items() if scene in names[fold::count]) for fold in range(count)]


def validate_partition(fitting_ids, validation_ids, test_ids, scenes):
    groups = [list(fitting_ids), list(validation_ids), list(test_ids)]
    if any(len(group) != len(set(group)) for group in groups):
        raise ValueError("Partition contains duplicate image IDs")
    fitting, validation, test = map(set, groups)
    if fitting & validation or fitting & test or validation & test:
        raise ValueError("Partition image IDs overlap")
    if any(index not in scenes for group in groups for index in group):
        raise ValueError("Partition has missing scene identities")
    scene_sets = [{scenes[index] for index in group} for group in groups]
    if scene_sets[0] & scene_sets[1] or scene_sets[0] & scene_sets[2] or scene_sets[1] & scene_sets[2]:
        raise ValueError("Partition scene identities overlap")


def fit_subset_scale(predictions, truths, fitting_ids):
    sampled_predictions = []
    sampled_truths = []
    for index in fitting_ids:
        prediction = predictions[index][45:471:8, 41:601:8]
        truth = truths[index][45:471:8, 41:601:8]
        valid = np.isfinite(truth) & (truth > 0.001) & (truth < 10)
        sampled_predictions.append(prediction[valid])
        sampled_truths.append(truth[valid])
    return fit_global_scale(sampled_predictions, sampled_truths)


def depth_boundary_band(truth, threshold=0.1, radius=3):
    truth = np.asarray(truth)
    valid = np.isfinite(truth) & (truth > 0.001) & (truth < 10)
    log_depth = np.log(np.where(valid, truth, 1.0))
    edges = np.zeros(truth.shape, np.uint8)
    horizontal = valid[:, 1:] & valid[:, :-1] & (np.abs(np.diff(log_depth, axis=1)) > threshold)
    vertical = valid[1:] & valid[:-1] & (np.abs(np.diff(log_depth, axis=0)) > threshold)
    edges[:, 1:] |= horizontal
    edges[:, :-1] |= horizontal
    edges[1:] |= vertical
    edges[:-1] |= vertical
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    band = cv2.dilate(edges, kernel).astype(bool) & valid
    crop = np.zeros_like(valid)
    crop[45:471, 41:601] = True
    return band & crop


def torch_nyu_metrics(prediction, truth):
    if prediction.shape != truth.shape or prediction.shape[-2:] != (480, 640):
        raise ValueError("NYU tensors require matching 480 by 640 predictions and truth")
    if not bool(torch.isfinite(prediction).all()):
        raise ValueError("Predictions must be finite")
    prediction = prediction.double().clamp(0.001, 10)[:, :, 45:471, 41:601]
    truth = truth.double()[:, :, 45:471, 41:601]
    valid = torch.isfinite(truth) & (truth > 0.001) & (truth < 10)
    count = valid.sum(dim=(1, 2, 3)).double()
    if not bool((count > 0).all()):
        raise ValueError("NYU images require valid evaluation pixels")
    safe_truth = torch.where(valid, truth, 1)
    error = torch.where(valid, prediction - safe_truth, 0)
    abs_rel = (error.abs() / safe_truth).sum(dim=(1, 2, 3)) / count
    rmse = (error.square().sum(dim=(1, 2, 3)) / count).sqrt()
    ratio = torch.maximum(prediction / safe_truth, safe_truth / prediction)
    delta = ((ratio < 1.25) & valid).sum(dim=(1, 2, 3)) / count
    return torch.stack([abs_rel, rmse, delta], dim=1)


def log_gradient_loss(prediction, truth, mask):
    prediction = prediction.clamp_min(0.001).log()
    truth = torch.where(mask, truth, 1).clamp_min(0.001).log()
    residual = prediction - truth
    losses = []
    for stride in (1, 2, 4):
        horizontal_valid = mask[..., stride:] & mask[..., :-stride]
        vertical_valid = mask[..., stride:, :] & mask[..., :-stride, :]
        horizontal = (residual[..., stride:] - residual[..., :-stride]).abs()
        vertical = (residual[..., stride:, :] - residual[..., :-stride, :]).abs()
        total = (horizontal * horizontal_valid).sum() + (vertical * vertical_valid).sum()
        count = horizontal_valid.sum() + vertical_valid.sum()
        losses.append(total / count.clamp_min(1))
    return torch.stack(losses).mean()


def validate_refit(fitting_ids, development_ids, test_ids, epochs, validation_ids):
    if set(fitting_ids) != set(development_ids) or len(fitting_ids) != len(set(fitting_ids)):
        raise ValueError("Refit must contain the complete unique development split")
    if set(fitting_ids) & set(test_ids):
        raise ValueError("Refit and test IDs overlap")
    if validation_ids:
        raise ValueError("Fixed-duration refit has no validation early stopping")
    if epochs < 1:
        raise ValueError("Learned refit requires a positive fixed duration")
