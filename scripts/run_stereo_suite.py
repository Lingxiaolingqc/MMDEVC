"""Tune only on declared dev scenes, or run/verify the frozen 15-pair baseline."""

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import cv2
import numpy as np
import yaml

from run_stereo_baseline import ROOT, checksum, read_yaml, repo_path, run, select_sample, write_csv, write_json
from p3_depth.data.middlebury import EXPECTED_SCENES, load_sample
from p3_depth.evaluation.stereo import evaluate_region
from p3_depth.models.stereo_sgbm import StereoConfig, predict_stereo


def rows_from(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def summary(rows):
    result = []
    for split in ("all_15_descriptive", "dev", "evaluation"):
        for region in ("all_gt", "nonoccluded_gt"):
            selected = [r for r in rows if r["region"] == region and
                        (split == "all_15_descriptive" or r["split"] == split)]
            if not selected:
                continue
            total = sum(int(r["gt_pixels"]) for r in selected)
            predicted = sum(int(r["valid_prediction_pixels"]) for r in selected)
            values = {"split": split, "region": region, "scenes": len(selected),
                      "gt_pixels": total, "valid_prediction_pixels": predicted,
                      "pooled_coverage": predicted / total,
                      "zero_coverage_scenes": sum(int(r["valid_prediction_pixels"]) == 0 for r in selected)}
            for key in ("bad1_fraction_invalid_as_bad", "bad2_fraction_invalid_as_bad"):
                values[f"pooled_{key}"] = sum(float(r[key]) * int(r["gt_pixels"]) for r in selected) / total
            for key in ("coverage", "disparity_mae_px_on_valid", "abs_rel_on_valid", "sq_rel_m_on_valid",
                        "rmse_m_on_valid", "rmse_log_on_valid", "delta1_on_valid", "delta2_on_valid", "delta3_on_valid"):
                finite = [float(r[key]) for r in selected if r[key] not in (None, "")]
                values[f"macro_{key}"] = float(np.mean(finite)) if finite else None
            result.append(values)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("tune", "all", "verify"), required=True)
    parser.add_argument("--output-dir", required=True, help="New run directory, or existing run for verify")
    parser.add_argument("--config", default="configs/stereo_sgbm_selected.yaml")
    parser.add_argument("--manifest", default="manifests/middlebury.csv")
    parser.add_argument("--paths-config", default="configs/paths.local.yaml")
    args = parser.parse_args()
    manifest, output = repo_path(args.manifest), repo_path(args.output_dir)
    settings = read_yaml(repo_path(args.paths_config))
    data_root = repo_path(settings["data_roots"]["middlebury"])
    rows = rows_from(manifest)
    if len(rows) != 15 or {r["sample_id"] for r in rows} != set(EXPECTED_SCENES):
        raise ValueError("Expected exactly the 15 Middlebury trainingQ scenes")
    if any(r["split"] not in {"dev", "evaluation"} for r in rows):
        raise ValueError("Complete all scene splits before running the suite")
    # Keep alternate lighting/exposure images of the same scene on one side.
    splits = {r["sample_id"]: r["split"] for r in rows}
    for a, b in (("Motorcycle", "MotorcycleE"), ("Piano", "PianoL"), ("Playtable", "PlaytableP")):
        if splits[a] != splits[b]:
            raise ValueError(f"Related scenes cross the split: {a}/{b}")
    if args.phase == "verify":
        import json
        metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        if checksum(manifest) != metadata["manifest_sha256"]:
            raise ValueError("Manifest changed since this run")
        for row in rows:
            sample_id = row["sample_id"]
            child = output / sample_id
            child_meta = json.loads((child / "metadata.json").read_text(encoding="utf-8"))
            for relative, expected in child_meta["artifacts_sha256"].items():
                if checksum(child / relative) != expected:
                    raise ValueError(f"Artifact hash mismatch: {sample_id}/{relative}")
            for source in child_meta["sources"]:
                if checksum(data_root / source["path"]) != source["sha256"]:
                    raise ValueError(f"Input hash mismatch: {source['path']}")
            paths, _ = select_sample(manifest, data_root, sample_id)
            sample = load_sample(paths)
            pred = np.load(child / f"predictions/{sample_id}.npy", allow_pickle=False)
            disp = np.load(child / f"disparities/{sample_id}.npy", allow_pickle=False)
            valid = np.load(child / f"masks/{sample_id}.npy", allow_pickle=False)
            if pred.dtype != np.float32 or valid.dtype != np.bool_ or not np.array_equal(np.isfinite(pred), valid):
                raise ValueError(f"Prediction contract failed: {sample_id}")
            actual = {name: evaluate_region(pred, disp, valid, sample.depth_m, sample.disparity_px, mask)
                      for name, mask in {"all_gt": sample.valid_gt,
                                         "nonoccluded_gt": sample.valid_gt & sample.nonoccluded}.items()}
            expected = json.loads((child / "metrics.json").read_text(encoding="utf-8"))
            if actual != expected:
                raise ValueError(f"Recomputed metrics differ: {sample_id}")
        combined = [r for row in rows for r in rows_from(output / row["sample_id"] / "metrics.csv")]
        if combined != rows_from(output / "per_scene.csv"):
            raise ValueError("Combined table differs from child runs")
        if summary(combined) != json.loads((output / "summary.json").read_text(encoding="utf-8")):
            raise ValueError("Summary differs from per-scene metrics")
        print("PASS: all 15 predictions, masks, source/artifact hashes, recomputed metrics and summaries")
        return
    if output.exists():
        raise FileExistsError(f"Will not overwrite {output}")
    output.mkdir(parents=True)
    metadata = {"code_commit": git("rev-parse", "HEAD"), "working_tree_status": git("status", "--porcelain"),
                "phase": args.phase, "manifest_sha256": checksum(manifest), "arguments": vars(args)}
    if args.phase == "tune":
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
        dev = [r for r in rows if r["split"] == "dev"]
        if not dev:
            raise ValueError("No dev scenes")
        scores, configs = [], {}
        for block in (5, 3, 7):
            config = StereoConfig(block_size=block)
            configs[block] = config
            for row in dev:
                sample_id = row["sample_id"]
                paths, _ = select_sample(manifest, data_root, sample_id)
                sample = load_sample(paths)
                start = perf_counter()
                pred = predict_stereo(sample.left_rgb, sample.right_rgb, sample.calibration, config)
                elapsed = perf_counter() - start
                score = evaluate_region(pred.depth_m, pred.disparity_px, pred.valid,
                                        sample.depth_m, sample.disparity_px, sample.valid_gt)
                folder = output / f"block{block}" / sample_id
                folder.mkdir(parents=True)
                np.save(folder / "depth_m.npy", pred.depth_m)
                np.save(folder / "disparity_px.npy", pred.disparity_px)
                np.save(folder / "valid.npy", pred.valid)
                write_json(folder / "metrics.json", score)
                scores.append({"sample_id": sample_id, "block_size": block,
                               "bad1_invalid_as_bad": score["bad1_fraction_invalid_as_bad"],
                               "coverage": score["coverage"], "prediction_seconds": elapsed})
        ranked = sorted(configs, key=lambda b: (
            np.mean([r["bad1_invalid_as_bad"] for r in scores if r["block_size"] == b]), abs(b - 5), b))
        selected = configs[ranked[0]]
        # This file is the only config emitted by tuning; copy it to args.config before the fixed run.
        (output / "selected.yaml").write_text(yaml.safe_dump(asdict(selected)), encoding="utf-8", newline="\n")
        write_csv(output / "candidates.csv", scores)
        metadata.update({"dev_scenes": [r["sample_id"] for r in dev], "candidate_block_sizes": [5, 3, 7],
                         "selection": "lowest unweighted dev-scene mean all-GT Bad>1px including invalids; tie: nearest block size to 5, then smaller",
                         "selected_block_size": selected.block_size})
        print(f"Selected block_size={selected.block_size} using only {[r['sample_id'] for r in dev]}")
    else:
        configuration = repo_path(args.config)
        original_hash = checksum(configuration)
        for row in rows:
            if checksum(configuration) != original_hash:
                raise ValueError("Frozen config changed during suite")
            run(argparse.Namespace(sample=row["sample_id"], config=args.config, manifest=args.manifest,
                                   paths_config=args.paths_config, data_root=None,
                                   output_dir=(output / row["sample_id"]).relative_to(ROOT).as_posix()))
        combined = [r for row in rows for r in rows_from(output / row["sample_id"] / "metrics.csv")]
        write_csv(output / "per_scene.csv", combined)
        aggregated = summary(combined)
        write_csv(output / "summary.csv", aggregated)
        write_json(output / "summary.json", aggregated)
        metadata["frozen_config_sha256"] = original_hash
        metadata["summary_policy"] = "Per-scene metrics macro averaged; coverage/bad also pixel pooled. All-15 includes dev, not held-out."
    write_json(output / "metadata.json", metadata)
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
