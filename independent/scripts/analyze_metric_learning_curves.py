"""Plot repeated-seed data-size curves using saved validation predictions."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors.torch import load_file
from torch.nn import functional as F

from depth_project.metric_research import torch_nyu_metrics
from depth_project.metric_training import load_development_pack, scaled_features, write_json
from depth_project.records import sha256_file
from depth_project.residual_refiner import ResidualRefiner


def evaluate_refiner(pack, summary, checkpoint, device="cuda"):
    if sha256_file(checkpoint) != summary["checkpoint_sha256"]:
        raise ValueError("Selected candidate checkpoint identity differs")
    config = summary["config"]
    positions = {index: position for position, index in enumerate(pack["ids"])}
    ids = config["validation_ids"]
    indices = [positions[index] for index in ids]
    base = np.maximum(pack["base"][indices] * summary["scale"], 0.001)
    features = scaled_features(pack["features"][indices], pack["base"][indices], summary["scale"]).astype(np.float16)
    model = ResidualRefiner(**config.get("model", {})).to(device).eval()
    model.load_state_dict(load_file(str(checkpoint), device=device))
    rows = []
    with torch.inference_mode():
        for start in range(0, len(ids), 8):
            end = min(start + 8, len(ids))
            values = torch.from_numpy(features[start:end]).to(device).float()
            frozen = None
            if config.get("model", {}).get("feature_channels"):
                frozen = torch.from_numpy(pack["depth_features"][indices[start:end]]).to(device)
            with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                residual = model(values, frozen)
            residual = F.interpolate(residual.float(), size=(480, 640), mode="bilinear", align_corners=False)
            prediction = torch.from_numpy(base[start:end, None]).to(device) * residual.exp()
            truth = torch.from_numpy(pack["truth"][indices[start:end], None]).to(device)
            scores = torch_nyu_metrics(prediction, truth).cpu().numpy()
            for index, score in zip(ids[start:end], scores):
                rows.append({"image_id": index, "scene": pack["scenes"][index], "abs_rel": float(score[0]),
                             "rmse": float(score[1]), "delta1": float(score[2])})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="curves")
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    pack = load_development_pack(root)
    jobs = json.loads((root / f"runs/metric-research/{args.stage}-jobs.json").read_text())
    summaries = []
    for job in jobs:
        folder = root / "runs/metric-research" / job["name"]
        summary = json.loads((folder / "summary.json").read_text())
        rows = evaluate_refiner(pack, summary, folder / "best.safetensors")
        replay = float(np.mean([row["abs_rel"] for row in rows]))
        if abs(replay - summary["best_abs_rel"]) > 1e-7:
            raise ValueError("Saved checkpoint does not reproduce its selection score")
        summaries.append({"name": job["name"], "target_image_count": job.get("target_image_count"),
                          "fit_image_count": len(job["fit_ids"]), "fit_scene_count": len({pack['scenes'][index] for index in job['fit_ids']}),
                          "seed": job["seed"], "abs_rel": replay, "best_step": summary["best_step"],
                          "completed_steps": summary["completed_steps"], "rows": rows})
        torch.cuda.empty_cache()
    result = {"runs": summaries, "selection_images": len(pack["validation_ids"]),
              "scope": "Repeated development selection; three seeds, source-scene-separated nested fitting subsets"}
    output = root / "results/metric-research"
    output.mkdir(exist_ok=True)
    if args.stage == "curves":
        targets = sorted({row["target_image_count"] for row in summaries})
        mean_errors = []
        actual_counts = []
        scene_counts = []
        deviations = []
        for target in targets:
            selected = [row for row in summaries if row["target_image_count"] == target]
            actual_counts.append(float(np.mean([row["fit_image_count"] for row in selected])))
            scene_counts.append(float(np.mean([row["fit_scene_count"] for row in selected])))
            mean_errors.append(float(np.mean([row["abs_rel"] for row in selected])))
            deviations.append(float(np.std([row["abs_rel"] for row in selected], ddof=1)))
        result["aggregate"] = {"targets": targets, "mean_actual_image_count": actual_counts,
                               "mean_scene_count": scene_counts, "mean_abs_rel": mean_errors, "seed_standard_deviation": deviations}
        ordered_ids = pack["validation_ids"]
        validation_scenes = sorted({pack["scenes"][index] for index in ordered_ids})
        clusters = [np.array([position for position, index in enumerate(ordered_ids) if pack["scenes"][index] == scene]) for scene in validation_scenes]
        sampled = np.random.default_rng(320113).integers(0, len(clusters), size=(5000, len(clusters)))
        comparisons = []
        for smaller, larger in zip(targets, targets[1:]):
            errors = []
            for target in [smaller, larger]:
                selected = [row for row in summaries if row["target_image_count"] == target]
                if any([row["image_id"] for row in run["rows"]] != ordered_ids for run in selected):
                    raise ValueError("Learning-curve paired validation identities differ")
                mean_per_image = np.mean([[row["abs_rel"] for row in run["rows"]] for run in selected], axis=0)
                errors.append(np.array([mean_per_image[cluster].sum() for cluster in clusters]))
            reduction = 100 * (1 - errors[1][sampled].sum(axis=1) / errors[0][sampled].sum(axis=1))
            comparisons.append({"smaller_target": smaller, "larger_target": larger,
                                "relative_abs_rel_reduction_percent": float(100 * (1 - errors[1].sum() / errors[0].sum())),
                                "scene_bootstrap_95_interval_percent": np.percentile(reduction, [2.5, 97.5]).tolist(),
                                "scope": "Paired scene uncertainty conditional on the three trained seeds"})
        result["adjacent_size_comparisons"] = comparisons
        figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for axis, counts, label in zip(axes, [actual_counts, scene_counts], ["Fitting images", "Fitting scenes"]):
            axis.errorbar(counts, mean_errors, yerr=deviations, color="#20566d", marker="o", capsize=4)
            axis.set(xlabel=label, ylabel="Validation AbsRel (lower is better)", title="Effect of training-set size")
            axis.grid(alpha=0.2)
        figure.suptitle("218 fixed validation images • mean and standard deviation across three seeds", fontsize=10)
        figure.savefig(output / "learning-curves.png", dpi=180)
        plt.close(figure)
    write_json(output / f"{args.stage}-analysis.json", result)
    print(json.dumps(result.get("aggregate", [{"name": row["name"], "abs_rel": row["abs_rel"]} for row in summaries]), indent=2))


if __name__ == "__main__":
    main()
