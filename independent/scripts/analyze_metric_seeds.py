"""Apply the fixed three-seed family and ensemble selection rules."""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_metric_folds import load_predictions
from depth_project.metric_selection import paired_scene_comparison, select_seed_ensemble
from depth_project.metric_training import load_development_pack, write_json
from depth_project.metrics import average_image_metrics, evaluate_nyu
from depth_project.records import run_provenance
from depth_project.refinement import blend_flip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-available", action="store_true", help="Compare inference variants sharing the already-trained seed recipes")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    folds = json.loads((root / "results/metric-research/scene-fold-selection.json").read_text())
    pack = load_development_pack(root)
    validation = pack["validation_ids"]
    positions = {index: position for position, index in enumerate(pack["ids"])}
    truth = pack["truth"][[positions[index] for index in validation]]
    results = []
    sources = {}
    eligible = [item["candidate"] for item in folds["ranking"]] if args.all_available else folds["advanced_candidates"]
    for candidate in ["control", *eligible]:
        if any(result["candidate"] == candidate for result in results):
            continue
        recipe = candidate.removesuffix("-flip").removesuffix("-feature-mean")
        predictions = {}
        seed_rows = []
        for seed in [320112, 320113, 320114]:
            run_name = f"selection-{recipe}-seed{seed}"
            summary, arrays, digest = load_predictions(root, run_name, pack["fit_ids"], validation)
            sources[run_name] = {"checkpoint_sha256": summary["checkpoint_sha256"], "predictions_sha256": digest,
                                 "scale": summary["scale"], "config": summary["config"]}
            prediction = arrays["depth_m"]
            if candidate.endswith("-flip"):
                prediction = blend_flip(prediction, arrays["flip_m"], 0.25)
            if candidate.endswith("-feature-mean"):
                feature_name = f"selection-features-gate-seed{seed}"
                feature, feature_arrays, feature_digest = load_predictions(root, feature_name, pack["fit_ids"], validation)
                sources[feature_name] = {"checkpoint_sha256": feature["checkpoint_sha256"],
                                         "predictions_sha256": feature_digest, "scale": feature["scale"], "config": feature["config"]}
                prediction = ((prediction.astype(np.float64) + feature_arrays["depth_m"]) / 2).astype(np.float32)
            predictions[seed] = prediction
            seed_rows.append([{**evaluate_nyu(value, target), "image_id": image_id, "scene": pack["scenes"][image_id]}
                              for image_id, value, target in zip(validation, prediction, truth)])
        seed_metrics = [average_image_metrics(rows) for rows in seed_rows]
        mean_rows = [{"image_id": image_id, "scene": pack["scenes"][image_id],
                      **{key: float(np.mean([rows[number][key] for rows in seed_rows])) for key in ["abs_rel", "rmse", "delta1"]}}
                     for number, image_id in enumerate(validation)]
        selection = select_seed_ensemble(predictions, truth)
        results.append({"candidate": candidate, "training_recipe": recipe, "seed_metrics": seed_metrics,
                        "mean_seed_metrics": average_image_metrics(mean_rows), "mean_seed_rows": mean_rows,
                        "seed_abs_rel_standard_deviation": float(np.std([item["abs_rel"] for item in seed_metrics], ddof=1)),
                        "ensemble_selection": selection})
        print(json.dumps({"candidate": candidate, "mean_abs_rel": results[-1]["mean_seed_metrics"]["abs_rel"],
                          "chosen_seeds": selection["chosen_seeds"]}), flush=True)
    ranking = sorted((item for item in results if item["candidate"] in eligible),
                     key=lambda item: (item["mean_seed_metrics"]["abs_rel"], item["candidate"]))
    control = next(item for item in results if item["candidate"] == "control")
    winner = ranking[0]
    result = {"scope": "577 fitting / 218 scene-disjoint selection images, three declared seeds, no test images",
              "rule": "Choose lowest three-seed mean AbsRel, then compare median single seed with the equal three-seed prediction mean",
              "winner": winner["candidate"], "chosen_seeds": winner["ensemble_selection"]["chosen_seeds"],
              "ranking": ranking, "control": control, "sources": sources,
              "winner_vs_control": paired_scene_comparison(control["mean_seed_rows"], winner["mean_seed_rows"]),
              "data_identities": pack["identities"], "provenance": run_provenance(root),
              "candidate_policy": "All inference variants of the trained scene-CV recipes" if args.all_available else "Two scene-CV finalists"}
    if len(ranking) > 1:
        result["winner_vs_runner_up"] = paired_scene_comparison(ranking[1]["mean_seed_rows"], winner["mean_seed_rows"])
    filename = "seed-variant-selection.json" if args.all_available else "seed-selection.json"
    write_json(root / "results/metric-research" / filename, result)


if __name__ == "__main__":
    main()
