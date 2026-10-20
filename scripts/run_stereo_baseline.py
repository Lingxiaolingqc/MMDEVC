"""Run one original-resolution Middlebury pair using CPU StereoSGBM, then evaluate."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
import yaml

from p3_depth.data.middlebury import MANIFEST_FIELDS, SamplePaths, load_sample
from p3_depth.evaluation.stereo import evaluate_region
from p3_depth.models.stereo_sgbm import StereoConfig, predict_stereo
from p3_depth.visualization.stereo import DISPLAY_RANGES, save_stereo_preview


def repo_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_yaml(path):
    result = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"YAML must be a mapping: {path}")
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def select_sample(manifest, data_root, sample_id):
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(MANIFEST_FIELDS):
            raise ValueError("Unexpected Middlebury manifest header")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Malformed manifest row")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate manifest sample IDs")
    selected = [row for row in rows if row["sample_id"] == sample_id]
    if len(selected) != 1:
        raise ValueError(f"Sample not found in manifest: {sample_id}")
    row = selected[0]
    if row["split"] not in {"dev", "evaluation"}:
        raise ValueError("Assign sample split before model development/evaluation")
    if not sample_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in sample_id):
        raise ValueError("Sample ID must be a safe filename")

    def data_path(key):
        relative = Path(row[key])
        result = (data_root / relative).resolve()
        if relative.is_absolute() or not result.is_relative_to(data_root):
            raise ValueError("Manifest paths must stay inside the data root")
        return result

    paths = SamplePaths(sample_id, data_path("left_path"), data_path("right_path"),
                        data_path("disparity_gt_path"), data_path("calibration_path"))
    if not paths.nonocclusion_mask.resolve().is_relative_to(data_root):
        raise ValueError("Mask path resolves outside the data root")
    return paths, row["split"]


def run(args):
    config_file, manifest = repo_path(args.config), repo_path(args.manifest)
    config = StereoConfig(**read_yaml(config_file))
    settings_file = repo_path(args.paths_config)
    settings = read_yaml(settings_file) if settings_file.exists() else {}
    roots = settings.get("data_roots", {})
    if not isinstance(roots, dict):
        raise ValueError("data_roots must be a mapping")
    root_value = args.data_root or roots.get("middlebury")
    if not root_value:
        raise ValueError("Create configs/paths.local.yaml or pass --data-root")
    data_root = repo_path(root_value)
    paths, split = select_sample(manifest, data_root, args.sample)
    sample = load_sample(paths)
    parameters = config.resolved(sample.calibration)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = repo_path(args.output_dir) if args.output_dir else (
        repo_path(settings.get("output_root", "outputs")) / f"stereo_{args.sample}_{stamp}")
    if output.exists():
        raise FileExistsError(f"Use a new output directory; will not overwrite {output}")
    output.mkdir(parents=True)
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    # Warm up identical code; timing covers two matchers + conversion + LR/depth, no disk/GT metrics.
    predict_stereo(sample.left_rgb, sample.right_rgb, sample.calibration, config)
    start = perf_counter()
    prediction = predict_stereo(sample.left_rgb, sample.right_rgb, sample.calibration, config)
    elapsed = perf_counter() - start
    metrics = {name: evaluate_region(prediction.depth_m, prediction.disparity_px, prediction.valid,
                                     sample.depth_m, sample.disparity_px, region)
               for name, region in {"all_gt": sample.valid_gt,
                                    "nonoccluded_gt": sample.valid_gt & sample.nonoccluded}.items()}
    for folder in ("predictions", "masks", "disparities", "diagnostics", "visualizations"):
        (output / folder).mkdir()
    saved = {
        f"predictions/{args.sample}.npy": prediction.depth_m,
        f"masks/{args.sample}.npy": prediction.valid,
        f"disparities/{args.sample}.npy": prediction.disparity_px,
        "diagnostics/left_fixed16.npy": prediction.left_fixed16,
        "diagnostics/right_fixed16.npy": prediction.right_fixed16,
        "diagnostics/left_before_lr_px.npy": prediction.left_disparity_px,
        "diagnostics/right_signed_px.npy": prediction.right_disparity_px,
        "diagnostics/lr_error_px.npy": prediction.lr_error_px,
        "diagnostics/evaluation_all_gt.npy": sample.valid_gt,
        "diagnostics/evaluation_nonoccluded_gt.npy": sample.valid_gt & sample.nonoccluded,
    }
    for relative, array in saved.items():
        np.save(output / relative, array, allow_pickle=False)
    write_csv(output / "predictions.csv", [{
        "sample_id": args.sample, "clip_id": "", "frame_index": "", "timestamp_s": "",
        "prediction_path": f"predictions/{args.sample}.npy", "mask_path": f"masks/{args.sample}.npy",
    }])
    rows = []
    for region, scores in metrics.items():
        row = {"sample_id": args.sample, "split": split, "region": region,
               **{key: value for key, value in scores.items() if key != "depth_on_valid_predictions"},
               **{f"{key}_on_valid": value for key, value in scores["depth_on_valid_predictions"].items()}}
        rows.append(row)
    write_csv(output / "metrics.csv", rows)
    write_json(output / "metrics.json", metrics)
    (output / "config_resolved.yaml").write_text(yaml.safe_dump(parameters, sort_keys=False), encoding="utf-8")
    display = {**DISPLAY_RANGES, "disparity_px": [0, parameters["num_disparities"]]}
    save_stereo_preview(sample, prediction, output / "visualizations" / f"{args.sample}.png", display)

    def git(*arguments):
        result = subprocess.run(["git", *arguments], cwd=ROOT, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError("Cannot capture Git provenance")
        return result.stdout.strip()

    sources = [paths.left, paths.right, paths.calibration, paths.disparity_gt, paths.nonocclusion_mask]
    implementations = [Path(__file__).resolve(), *sorted((ROOT / "src/p3_depth").rglob("*.py"))]
    log = (f"PASS {args.sample} ({split}): {sample.calibration.width}x{sample.calibration.height}\n"
           f"CPU prediction: {elapsed:.6f} s; 1 warmup + 1 measured pair, not a benchmark\n"
           f"All-GT coverage: {metrics['all_gt']['coverage']:.6%}\n"
           f"All-GT RMSE on valid predictions only: {metrics['all_gt']['depth_on_valid_predictions']['rmse_m']} m\n"
           f"All-GT bad >1px, invalid counted as bad: {metrics['all_gt']['bad1_fraction_invalid_as_bad']:.6%}\n")
    (output / "run.log").write_text(log, encoding="utf-8")
    metadata = {
        "schema_version": 1, "status": "completed", "kind": "single_pair_development_diagnostic" if split == "dev" else "single_pair_evaluation",
        "run_id": output.name, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": git("rev-parse", "HEAD"), "working_tree_status": git("status", "--porcelain"),
        "dataset": "Middlebury v3 trainingQ", "sample_id": args.sample, "split": split,
        "manifest_sha256": checksum(manifest), "config_sha256": checksum(config_file),
        "checkpoint": None, "official_code": f"OpenCV {cv2.__version__} installed binary; no third-party checkout",
        "prediction_type": "metric_depth", "unit": "m", "depth_axis": "camera Z",
        "disparity_convention": "left: x_left-x_right; right: x_right-x_left; raw int16 divided by 16",
        "depth_formula": "f_px * baseline_mm / (1000 * (disparity_px + doffs_px))",
        "alignment": "none", "resize_crop": "none", "input_output_shape_hw": list(prediction.depth_m.shape),
        "calibration": {"focal_px": sample.calibration.focal_px, "baseline_mm": sample.calibration.baseline_mm,
                        "doffs_px": sample.calibration.doffs_px, "ndisp": sample.calibration.ndisp},
        "prediction_mask": "left/right finite matches; in-bounds correspondence; LR residual <= threshold; positive finite Z",
        "evaluation": {"regions": "fixed GT masks: labels != 0 and labels == 255; no crop/depth clipping",
                       "invalid_policy": "coverage reported; invalids count as bad1/bad2; depth errors conditional on valid predictions",
                       "rmse_log_base": "natural logarithm", "bad_threshold_unit": "Q-resolution pixels; strict >; project diagnostic, not leaderboard submission"},
        "coverage_image_before_lr": float(np.isfinite(prediction.left_disparity_px).mean()),
        "coverage_image_final": float(prediction.valid.mean()), "display_ranges": display,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "processor": platform.processor(), "device": "CPU", "opencv_threads": cv2.getNumThreads(),
                        "opencl": cv2.ocl.useOpenCL(), "opencv": cv2.__version__,
                        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "Pillow", "PyYAML", "matplotlib")}},
        "timing": {"prediction_seconds": elapsed, "warmup_pairs": 1, "measured_pairs": 1,
                   "scope": "grayscale conversion, matcher creation, left/right matching, LR check and depth conversion; excludes IO/GT metrics/plots",
                   "benchmark": False},
        "sources": [{"path": path.relative_to(data_root).as_posix(), "sha256": checksum(path)} for path in sources],
        "implementation_sha256": {path.relative_to(ROOT).as_posix(): checksum(path) for path in implementations},
        "artifacts_sha256": {path.relative_to(output).as_posix(): checksum(path) for path in sorted(output.rglob("*")) if path.is_file()},
        "arguments": vars(args),
        "reproduce": ["python", "scripts/run_stereo_baseline.py", *sys.argv[1:]],
    }
    write_json(output / "metadata.json", metadata)
    print(log, end="")
    print(f"Output: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", default="Adirondack")
    parser.add_argument("--config", default="configs/stereo_sgbm.yaml")
    parser.add_argument("--manifest", default="manifests/middlebury.csv")
    parser.add_argument("--paths-config", default="configs/paths.local.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir", help="New directory; existing output is never overwritten")
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, TypeError, RuntimeError, yaml.YAMLError, cv2.error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
