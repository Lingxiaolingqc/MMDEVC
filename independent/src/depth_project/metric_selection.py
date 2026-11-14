"""Predeclared development selection and paired scene-level comparisons."""
import math

import numpy as np

from .metrics import evaluate_nyu


def rank_fold_results(records):
    """Rank candidates by the equal mean of three scene-fold evaluations."""
    candidates = {}
    for record in records:
        candidates.setdefault(record["candidate"], []).append(record)
    ranking = []
    for name, folds in candidates.items():
        if sorted(record["fold"] for record in folds) != [0, 1, 2]:
            raise ValueError("Each candidate requires exactly three distinct scene folds")
        folds = sorted(folds, key=lambda record: record["fold"])
        metrics = np.array([[record["metrics"][key] for key in ("abs_rel", "rmse", "delta1")]
                            for record in folds], dtype=float)
        if not np.isfinite(metrics).all() or (metrics < 0).any():
            raise ValueError("Fold metrics must be finite and nonnegative")
        ranking.append({"candidate": name, "folds": folds,
                        "mean_fold_metrics": dict(zip(("abs_rel", "rmse", "delta1"),
                                                       metrics.mean(axis=0).tolist()))})
    return sorted(ranking, key=lambda item: (item["mean_fold_metrics"]["abs_rel"], item["candidate"]))


def fixed_refit_epochs(best_epochs):
    values = np.asarray(best_epochs, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Refit duration requires three finite, nonnegative fold checkpoint epochs")
    return math.floor(float(np.median(values)) + 0.5)


def select_seed_ensemble(predictions, truth):
    if len(predictions) != 3:
        raise ValueError("Seed selection requires exactly three trained seeds")
    truth = np.asarray(truth)
    if truth.ndim != 3:
        raise ValueError("Seed selection requires NHW ground truth")
    scores = {}
    ensemble = np.zeros(truth.shape, dtype=np.float64)
    for seed, values in sorted(predictions.items()):
        values = np.asarray(values)
        if values.shape != truth.shape or not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Seed predictions must have matching shape and finite positive metric depth")
        scores[seed] = float(np.mean([evaluate_nyu(prediction, target)["abs_rel"]
                                     for prediction, target in zip(values, truth)]))
        ensemble += values
    ensemble = (ensemble / 3).astype(np.float32)
    median_score = float(np.median(list(scores.values())))
    median_seed = min(seed for seed, score in scores.items() if score == median_score)
    ensemble_score = float(np.mean([evaluate_nyu(prediction, target)["abs_rel"]
                                    for prediction, target in zip(ensemble, truth)]))
    chosen = sorted(predictions) if ensemble_score < median_score else [median_seed]
    return {"seed_abs_rel": scores, "median_seed": median_seed, "median_single_abs_rel": median_score,
            "ensemble_abs_rel": ensemble_score, "chosen_seeds": chosen}


def paired_scene_comparison(before, after, seed=320112, samples=5000):
    identities = [(row["image_id"], row["scene"]) for row in before]
    if not identities or len({item[0] for item in identities}) != len(identities) or identities != [(row["image_id"], row["scene"]) for row in after]:
        raise ValueError("Scene comparison requires unique paired image and scene identities")
    original = np.array([row["abs_rel"] for row in before])
    improved = np.array([row["abs_rel"] for row in after])
    if not np.isfinite(original).all() or not np.isfinite(improved).all() or (original < 0).any() or (improved < 0).any() or original.sum() <= 0:
        raise ValueError("Scene comparison requires finite nonnegative errors and positive reference error")
    scenes = sorted({scene for _, scene in identities})
    membership = np.array([scene for _, scene in identities])
    original_sums = np.array([original[membership == scene].sum() for scene in scenes])
    improved_sums = np.array([improved[membership == scene].sum() for scene in scenes])
    sampled = np.random.default_rng(seed).integers(0, len(scenes), (samples, len(scenes)))
    reduction = 100 * (1 - improved_sums[sampled].sum(axis=1) / original_sums[sampled].sum(axis=1))
    return {"relative_abs_rel_reduction_percent": float(100 * (1 - improved.mean() / original.mean())),
            "scene_bootstrap_95_interval_percent": np.percentile(reduction, [2.5, 97.5]).tolist(),
            "image_count": len(before), "scene_count": len(scenes),
            "improved_images": int((improved < original).sum()), "worsened_images": int((improved > original).sum()),
            "scope": "Paired scene uncertainty conditional on the evaluated fitted models"}
