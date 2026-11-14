"""Compare predeclared inference variants on three fitting-scene folds."""
import json
from pathlib import Path

import numpy as np

from depth_project.metric_research import scene_folds
from depth_project.metric_selection import fixed_refit_epochs, rank_fold_results
from depth_project.metric_training import load_development_pack, write_json
from depth_project.metrics import average_image_metrics, evaluate_nyu
from depth_project.records import run_provenance, sha256_file
from depth_project.refinement import blend_flip


def load_predictions(root, name, fitting, validation):
    folder = root / "runs/metric-research" / name
    summary = json.loads((folder / "summary.json").read_text())
    config = summary["config"]
    if config["fit_ids"] != fitting or config["validation_ids"] != validation:
        raise ValueError(f"Scene-fold partition differs for {name}")
    if sha256_file(folder / "best.safetensors") != summary["checkpoint_sha256"]:
        raise ValueError(f"Scene-fold checkpoint differs for {name}")
    path = folder / "validation.npz"
    packet_hash = sha256_file(path)
    if summary.get("validation_predictions_sha256") and packet_hash != summary["validation_predictions_sha256"]:
        raise ValueError(f"Scene-fold prediction identity differs for {name}")
    with np.load(path, allow_pickle=False) as packet:
        if packet["image_ids"].tolist() != validation:
            raise ValueError(f"Scene-fold prediction IDs differ for {name}")
        arrays = {key: packet[key] for key in ("depth_m", "flip_m") if key in packet}
    for values in arrays.values():
        if values.shape != (len(validation), 480, 640) or not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"Scene-fold metric predictions are invalid for {name}")
    return summary, arrays, packet_hash


def main():
    root = Path(__file__).resolve().parents[1]
    pack = load_development_pack(root)
    positions = {index: position for position, index in enumerate(pack["ids"])}
    folds = scene_folds({index: pack["scenes"][index] for index in pack["fit_ids"]}, 3, 320113)
    records = []
    sources = {}
    for fold, validation in enumerate(folds):
        fitting = sorted(set(pack["fit_ids"]) - set(validation))
        predictions = {}
        summaries = {}
        for name in ["control", "features-gate", "lora-r8-high", "lora-r8-low"]:
            run_name = f"{name}-fold{fold}"
            summary, arrays, digest = load_predictions(root, run_name, fitting, validation)
            summaries[name] = summary
            predictions[name] = arrays["depth_m"]
            sources[run_name] = {"checkpoint_sha256": summary["checkpoint_sha256"],
                                 "predictions_sha256": digest, "best_epoch": summary["best_epoch"]}
            if name.startswith("lora"):
                predictions[f"{name}-flip"] = blend_flip(arrays["depth_m"], arrays["flip_m"], 0.25)
                predictions[f"{name}-feature-mean"] = ((arrays["depth_m"].astype(np.float64)
                                                        + predictions["features-gate"]) / 2).astype(np.float32)
        for name, values in predictions.items():
            rows = []
            for image_id, prediction in zip(validation, values):
                rows.append({"image_id": image_id, "scene": pack["scenes"][image_id],
                             **evaluate_nyu(prediction, pack["truth"][positions[image_id]])})
            metrics = average_image_metrics(rows)
            recipe = name.removesuffix("-flip").removesuffix("-feature-mean")
            best_epoch = summaries[recipe]["best_epoch"]
            if name == recipe and abs(metrics["abs_rel"] - summaries[recipe]["best_abs_rel"]) > 1e-7:
                raise ValueError(f"Saved scene-fold predictions do not replay {name}")
            records.append({"candidate": name, "fold": fold, "training_recipe": recipe,
                            "best_epoch": best_epoch, "metrics": metrics, "rows": rows})
            print(json.dumps({"candidate": name, "fold": fold, "abs_rel": metrics["abs_rel"]}), flush=True)
    ranking = rank_fold_results(records)
    for candidate in ranking:
        candidate["fixed_refit_epochs"] = fixed_refit_epochs([fold["best_epoch"] for fold in candidate["folds"]])
    result = {"scope": "Three scene folds inside 577 fitting images; equal fold means; no test images",
              "selection_metric": "Lowest mean full-resolution course-protocol AbsRel over all three folds",
              "ranking": ranking, "advanced_candidates": [item["candidate"] for item in ranking[:2]],
              "sources": sources, "data_identities": pack["identities"], "provenance": run_provenance(root)}
    write_json(root / "results/metric-research/scene-fold-selection.json", result)


if __name__ == "__main__":
    main()
