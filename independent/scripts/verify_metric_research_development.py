"""Verify frozen RGB-only inference against the saved development selection."""
import json
from pathlib import Path

import h5py
import numpy as np

from depth_project.research_inference import load_research_predictor
from depth_project.metric_training import write_json
from depth_project.metrics import average_image_metrics, evaluate_nyu
from depth_project.records import run_provenance, sha256_file
from depth_project.refinement import blend_flip
from depth_project.research_evaluation import verify_recipe


def main():
    root = Path(__file__).resolve().parents[1]
    path = root / "configs/metric-research-final.json"
    config = json.loads(path.read_text())
    verify_recipe(root, config)
    selection = json.loads((root / "results/metric-research/seed-selection.json").read_text())
    winner = config["winner"]
    recipe = winner.removesuffix("-flip").removesuffix("-feature-mean")
    combined = config["pre_refit"].get("type") == "lora-feature-mean"
    specification = config["pre_refit"]["lora"] if combined else config["pre_refit"]
    validation = config["selection_validation_ids"]
    saved = []
    for member in specification["members"]:
        folder = root / "runs/metric-research" / f"selection-{recipe}-seed{member['seed']}"
        with np.load(folder / "validation.npz", allow_pickle=False) as packet:
            if packet["image_ids"].tolist() != validation:
                raise ValueError("Saved selection prediction identities differ")
            prediction = packet["depth_m"]
            if specification["flip_weight"]:
                prediction = blend_flip(prediction, packet["flip_m"], specification["flip_weight"])
            if combined:
                feature_path = root / "runs/metric-research" / f"selection-features-gate-seed{member['seed']}/validation.npz"
                with np.load(feature_path, allow_pickle=False) as feature_packet:
                    if feature_packet["image_ids"].tolist() != validation:
                        raise ValueError("Saved feature selection identities differ")
                    prediction = ((prediction.astype(np.float64) + feature_packet["depth_m"]) / 2).astype(np.float32)
            saved.append(prediction)
    expected = np.mean(np.stack(saved), axis=0, dtype=np.float64).astype(np.float32)
    predictor = load_research_predictor(root, config["pre_refit"], config["foundation_checkpoint_sha256"])
    maximum_difference = 0.0
    rows = []
    with h5py.File(root / "data/nyu/nyu_depth_v2_labeled.mat") as dataset:
        for position, index in enumerate(validation):
            rgb = np.ascontiguousarray(dataset["images"][index].transpose(2, 1, 0))
            prediction = predictor.predict_rgb(rgb).values
            maximum_difference = max(maximum_difference, float(np.max(np.abs(prediction - expected[position]))))
            rows.append({"image_id": index, **evaluate_nyu(prediction, dataset["depths"][index].T)})
    metrics = average_image_metrics(rows)
    selected = next(item for item in selection["ranking"] if item["candidate"] == winner)["ensemble_selection"]
    score = selected["ensemble_abs_rel"] if len(config["chosen_seeds"]) == 3 else selected["median_single_abs_rel"]
    result = {"metrics": metrics, "selection_abs_rel": score, "maximum_saved_prediction_difference_m": maximum_difference,
              "image_ids": validation, "configuration_sha256": sha256_file(path), "metadata": predictor.metadata,
              "scope": "Actual frozen pre-refit predictor, RGB-only input, all 218 scene-disjoint selection images",
              "provenance": run_provenance(root)}
    write_json(root / "results/metric-research/normal-entry-development-replay.json", result)
    print(json.dumps({"metrics": metrics, "maximum_saved_prediction_difference_m": maximum_difference}, indent=2))
    if maximum_difference > 1e-5 or abs(metrics["abs_rel"] - score) > 1e-7:
        raise ValueError("Frozen RGB-only inference does not reproduce development selection")


if __name__ == "__main__":
    main()
