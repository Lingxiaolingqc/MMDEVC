"""Compare small correction networks conditioned on frozen UniDepth features."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from depth_project.metric_training import load_development_pack, train_candidate, write_json
from depth_project.records import sha256_file


def attach_depth_features(root, pack):
    source = root / "runs/metric-research/foundation-default"
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("completed") != len(pack["ids"]) or manifest["identities"]["ids"] != pack["ids"]:
        raise ValueError("Complete the frozen feature cache before refiner training")
    for name in ["dataset_sha256", "split_sha256", "checkpoint_sha256"]:
        if manifest["identities"][name] != pack["identities"][name]:
            raise ValueError("Frozen feature source identities differ")
    if manifest["processed_hw"] != [490, 644]:
        raise ValueError("This feature cache expects the recorded default NYU preprocessing")
    path = root / ".cache/metric-research/depth-features.npy"
    complete = path.with_suffix(".json")
    if complete.exists():
        if json.loads(complete.read_text())["identities"] != manifest["identities"]:
            raise ValueError("Prepared frozen feature identities differ")
    else:
        temporary = path.with_suffix(".pending.npy")
        values = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16,
                                          shape=(len(pack["ids"]), *manifest["feature_shape"]))
        for position, index in enumerate(pack["ids"]):
            with np.load(source / f"prediction_{index:04d}.npz", allow_pickle=False) as packet:
                feature = packet["depth_features"]
                if int(packet["image_id"]) != index or not np.isfinite(feature).all():
                    raise ValueError("Frozen feature image or finite-value check failed")
                values[position] = feature
        values.flush()
        del values
        temporary.replace(path)
        write_json(complete, {"identities": manifest["identities"]})
    pack = {**pack, "depth_features": np.load(path, mmap_mode="r"),
            "identities": {**pack["identities"], "depth_features_sha256": sha256_file(path)}}
    return pack


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
    }
    summaries = []
    jobs = []
    for name, options in variants.items():
        model = {**base.get("model", {}), "feature_channels": pack["depth_features"].shape[1]}
        if options.get("gated"):
            model["gated"] = True
        jobs.append({"name": name, "fit_ids": pack["fit_ids"], "validation_ids": pack["validation_ids"],
                     "seed": 320112, "max_epochs": 80, "patience": 10, "batch_size": 8,
                     **base, **{key: value for key, value in options.items() if key != "gated"}, "model": model})
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
