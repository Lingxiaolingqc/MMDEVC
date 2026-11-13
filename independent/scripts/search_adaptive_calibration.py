"""Cross-validate a small image-specific metric scale regressor."""
import json
from pathlib import Path

import numpy as np

from depth_project.adaptive_calibration import fit_ridge_scale, global_features, predict_ridge_scale
from depth_project.metric_research import fit_subset_scale, scene_folds
from depth_project.metric_training import load_development_pack, write_json
from depth_project.metrics import evaluate_nyu


def main():
    root = Path(__file__).resolve().parents[1]
    pack = load_development_pack(root)
    source = root / "runs/metric-research/foundation-default"
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["completed"] != len(pack["ids"]) or manifest["identities"]["ids"] != pack["ids"]:
        raise ValueError("Complete all frozen visual features before calibration search")
    positions = {index: position for position, index in enumerate(pack["ids"])}
    features = []
    for position, index in enumerate(pack["ids"]):
        with np.load(source / f"prediction_{index:04d}.npz") as packet:
            if int(packet["image_id"]) != index:
                raise ValueError("Calibration feature image identity differs")
            features.append(global_features(packet["depth_features"], pack["base"][position]))
    features = np.stack(features)
    targets = {}
    for index in pack["fit_ids"]:
        position = positions[index]
        targets[index] = fit_subset_scale({index: pack["base"][position]}, {index: pack["truth"][position]}, [index])
    folds = scene_folds({index: pack["scenes"][index] for index in pack["fit_ids"]}, 3, 320113)

    def fit(indices, alpha):
        selected_features = features[[positions[index] for index in indices]]
        model = fit_ridge_scale(selected_features, [targets[index] for index in indices], alpha)
        predicted_scales = predict_ridge_scale(model, selected_features)
        adjusted = {index: pack["base"][positions[index]] * float(scale) for index, scale in zip(indices, predicted_scales)}
        model["metric_global_scale"] = fit_subset_scale(adjusted, {index: pack["truth"][positions[index]] for index in indices}, indices)
        return model

    candidates = []
    for alpha in [0.1, 1, 10, 100, 1000]:
        scores = []
        for held_out in folds:
            fitting = sorted(set(pack["fit_ids"]) - set(held_out))
            model = fit(fitting, alpha)
            scales = predict_ridge_scale(model, features[[positions[index] for index in held_out]])
            fold_rows = [evaluate_nyu(pack["base"][positions[index]] * float(scale), pack["truth"][positions[index]]) for index, scale in zip(held_out, scales)]
            scores.append(float(np.mean([row["abs_rel"] for row in fold_rows])))
        candidate = {"alpha": alpha, "fold_abs_rel": scores, "mean_abs_rel": float(np.mean(scores))}
        candidates.append(candidate)
        print(json.dumps(candidate), flush=True)
    selected = min(candidates, key=lambda row: row["mean_abs_rel"])
    fitting = pack["fit_ids"]
    model = fit(fitting, selected["alpha"])
    validation = pack["validation_ids"]
    scales = predict_ridge_scale(model, features[[positions[index] for index in validation]])
    rows = [{"image_id": index, "predicted_scale": float(scale), **evaluate_nyu(pack["base"][positions[index]] * float(scale), pack["truth"][positions[index]])} for index, scale in zip(validation, scales)]
    result = {"type": "adaptive_global_calibration", "fit_ids": fitting, "validation_ids": validation,
              "candidates": candidates, "selected_alpha": selected["alpha"],
              "selection_abs_rel": float(np.mean([row["abs_rel"] for row in rows])), "rows": rows,
              "model": {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in model.items()},
              "setting": "Scale predicted from RGB-only frozen features; oracle targets used solely for fitting images"}
    output = root / "runs/metric-research/adaptive-calibration"
    output.mkdir(exist_ok=True)
    write_json(output / "summary.json", result)
    print(json.dumps({"selection_abs_rel": result["selection_abs_rel"], "selected_alpha": selected["alpha"]}))


if __name__ == "__main__":
    main()
