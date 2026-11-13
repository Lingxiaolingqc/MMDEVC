"""Measure scale, boundaries and error ranking on development scenes."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from depth_project.metric_research import depth_boundary_band, fit_subset_scale, scene_folds
from depth_project.metric_training import load_development_pack, write_json
from depth_project.metrics import evaluate_nyu


def main():
    root = Path(__file__).resolve().parents[1]
    pack = load_development_pack(root)
    output = root / "runs/metric-research"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    fitting = set(pack["fit_ids"])
    for position, index in enumerate(pack["ids"]):
        truth = pack["truth"][position]
        with np.load(root / f"runs/nyu-development/prediction_{index:04d}.npz", allow_pickle=False) as sample:
            original = sample["depth_m"]
            error = np.abs(np.clip(original, 0.001, 10) - truth) / np.maximum(truth, 0.001)
            valid = np.isfinite(truth) & (truth > 0.001) & (truth < 10)
            crop = np.zeros_like(valid)
            crop[45:471, 41:601] = True
            valid &= crop
            boundary = depth_boundary_band(truth)
            sample_truth = truth[45:471:8, 41:601:8]
            sample_depth = original[45:471:8, 41:601:8]
            support = np.isfinite(sample_truth) & (sample_truth > 0.001) & (sample_truth < 10)
            log_error = np.abs(np.log(np.maximum(sample_depth[support], 0.001)) - np.log(sample_truth[support]))
            confidence = sample["confidence"][45:471:8, 41:601:8][support]
            disagreement = np.abs(original - sample["flip_m"])[45:471:8, 41:601:8][support]
            groups = {"near_under_2m": valid & (truth < 2), "mid_2_to_4m": valid & (truth >= 2) & (truth < 4),
                      "far_4_to_10m": valid & (truth >= 4), "boundary": valid & boundary, "interior": valid & ~boundary}
            conditional = {name: float(error[mask].mean()) if mask.any() else None for name, mask in groups.items()}
            row = {"image_id": index, "scene": pack["scenes"][index], "partition": "fitting" if index in fitting else "selection",
                   **evaluate_nyu(original, truth), "conditional_abs_rel": conditional,
                   "confidence_error_spearman": float(spearmanr(confidence, log_error).statistic),
                   "flip_disagreement_error_spearman": float(spearmanr(disagreement, log_error).statistic)}
            if index in fitting:
                row["diagnostic_oracle_scale"] = fit_subset_scale({index: original}, {index: truth}, [index])
            rows.append(row)
    groups = {}
    for partition in ["fitting", "selection"]:
        selected = [row for row in rows if row["partition"] == partition]
        groups[partition] = {"image_count": len(selected), "scene_count": len({row['scene'] for row in selected}),
                             "mean_abs_rel": float(np.mean([row["abs_rel"] for row in selected])),
                             "mean_conditional_abs_rel": {name: float(np.mean([row["conditional_abs_rel"][name] for row in selected if row["conditional_abs_rel"][name] is not None])) for name in selected[0]["conditional_abs_rel"]},
                             "median_confidence_error_spearman": float(np.median([row["confidence_error_spearman"] for row in selected])),
                             "median_disagreement_error_spearman": float(np.median([row["flip_disagreement_error_spearman"] for row in selected]))}
    scales = [row["diagnostic_oracle_scale"] for row in rows if "diagnostic_oracle_scale" in row]
    summary = {"groups": groups, "fitting_oracle_scale_quantiles": np.percentile(scales, [5, 25, 50, 75, 95]).tolist(),
               "scope": "Development-only diagnostic; per-image oracle scale is never an inference input", "rows": rows,
               "folds": scene_folds({index: pack["scenes"][index] for index in fitting}, 3, 320113),
               "identities": pack["identities"]}
    write_json(output / "development-diagnostics.json", summary)
    tracked = root / "results/metric-research"
    tracked.mkdir(exist_ok=True)
    write_json(tracked / "development-diagnostics.json", summary)
    print(json.dumps({"groups": groups, "scale_quantiles": summary["fitting_oracle_scale_quantiles"]}, indent=2))


if __name__ == "__main__":
    main()
