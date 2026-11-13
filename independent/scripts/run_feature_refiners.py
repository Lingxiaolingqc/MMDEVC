"""Compare small correction networks conditioned on frozen UniDepth features."""
import argparse
import json
from pathlib import Path

import torch

from depth_project.metric_feature_cache import attach_depth_features
from depth_project.metric_training import load_development_pack, train_candidate, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-run", help="Run name supplying selected augmentation/model-width settings")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    pack = attach_depth_features(root, load_development_pack(root))
    base = {}
    if args.base_run:
        previous = json.loads((root / "runs/metric-research" / args.base_run / "summary.json").read_text())["config"]
        base = {key: previous[key] for key in ["augment_rgb", "model"] if key in previous}
    variants = {
        "features-low": {},
        "features-full": {"full_resolution": True},
        "features-gradient": {"full_resolution": True, "gradient_weight": 0.05},
        "features-gate": {"full_resolution": True, "gradient_weight": 0.05, "gated": True},
        "features-bound1": {"residual_bound": 1.0},
    }
    summaries = []
    jobs = []
    for name, options in variants.items():
        model = {**base.get("model", {}), "feature_channels": pack["depth_features"].shape[1]}
        if options.get("gated"):
            model["gated"] = True
        if "residual_bound" in options:
            model["residual_bound"] = options["residual_bound"]
        jobs.append({"name": name, "fit_ids": pack["fit_ids"], "validation_ids": pack["validation_ids"],
                     "seed": 320112, "max_epochs": 80, "patience": 10, "batch_size": 8,
                     **base, **{key: value for key, value in options.items() if key not in {"gated", "residual_bound"}}, "model": model})
    output = root / "runs/metric-research"
    path = output / "feature-jobs.json"
    if path.exists() and json.loads(path.read_text()) != jobs:
        raise ValueError("Feature job manifest differs")
    write_json(path, jobs)
    for job in jobs:
        summary = train_candidate(pack, job, output / job["name"])
        summaries.append({"name": job["name"], "best_abs_rel": summary["best_abs_rel"], "best_epoch": summary["best_epoch"]})
        write_json(output / "feature-summary.json", summaries)
        torch.cuda.empty_cache()
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
