"""Evaluate one saved original-size prediction with explicit representation and units."""

import argparse
import csv
import subprocess

import numpy as np

from run_stereo_baseline import ROOT, checksum, read_yaml, repo_path, select_sample, write_csv, write_json
from p3_depth.data.middlebury import load_sample
from p3_depth.data.nyuv2 import evaluation_mask
from p3_depth.evaluation.depth import evaluate_depth, prepare_prediction


def ground_truth(dataset, sample_id, settings, manifest):
    data_root = repo_path(settings["data_roots"][dataset])
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["sample_id"] == sample_id]
    if len(rows) != 1:
        raise ValueError("Expected exactly one matching sample in manifest")
    row = rows[0]
    if dataset == "middlebury":
        paths, _ = select_sample(manifest, data_root, sample_id)
        sample = load_sample(paths)
        return sample.depth_m, {"all_gt": sample.valid_gt, "nonoccluded_gt": sample.valid_gt & sample.nonoccluded}, row, [paths.disparity_gt, paths.calibration, paths.nonocclusion_mask]
    gt_path = (data_root / row["depth_gt_path"]).resolve()
    if not gt_path.is_relative_to(data_root):
        raise ValueError("GT path escapes data root")
    gt = np.load(gt_path, allow_pickle=False)
    return gt, {"nyuv2_crop": evaluation_mask(gt)}, row, [gt_path]


def evaluate_saved(args):
    manifest = repo_path(args.manifest or f"manifests/{args.dataset}.csv")
    settings = read_yaml(repo_path(args.paths_config))
    gt, regions, row, sources = ground_truth(args.dataset, args.sample, settings, manifest)
    raw = np.load(repo_path(args.prediction), allow_pickle=False)
    supplied = np.load(repo_path(args.mask), allow_pickle=False) if args.mask else None
    # Fit once on the declared primary GT region, then reuse the same transformed prediction everywhere.
    primary = regions["all_gt"] if "all_gt" in regions else regions["nyuv2_crop"]
    depth, valid, transform = prepare_prediction(raw, gt, primary, args.prediction_type, args.unit, args.alignment, supplied)
    scores = {name: evaluate_depth(depth, gt, mask, valid) for name, mask in regions.items()}
    output = repo_path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Will not overwrite {output}")
    output.mkdir(parents=True)
    np.save(output / "evaluated_depth_m.npy", depth)
    np.save(output / "valid_prediction.npy", valid)
    write_json(output / "metrics.json", scores)
    write_csv(output / "metrics.csv", [{"sample_id": args.sample, "split": row["split"], "region": name,
        "result_label": transform["result_label"], "coverage": score["coverage"],
        **{f"{key}_on_valid": value for key, value in score["depth_on_valid_predictions"].items()},
        **{f"{key}_full_gt": value for key, value in score["delta_full_gt_invalid_as_failure"].items()}}
        for name, score in scores.items()])
    metadata = {"dataset": args.dataset, "sample_id": args.sample, "split": row["split"],
                "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "arguments": vars(args), "transformation": transform, "shape_hw": list(depth.shape),
                "resizing": "none; input must match original GT shape", "prediction_clipping": "none",
                "invalid_policy": "fixed GT denominator; conditional errors plus coverage and full-GT delta with invalids as failures",
                "manifest_sha256": checksum(manifest), "input_prediction_sha256": checksum(repo_path(args.prediction)),
                "input_mask_sha256": checksum(repo_path(args.mask)) if args.mask else None,
                "gt_sha256": {p.name: checksum(p) for p in sources},
                "raw_prediction_modified": False}
    write_json(output / "metadata.json", metadata)
    print(f"PASS {args.sample}: {transform['result_label']}; "
          + "; ".join(f"{name}: coverage={score['coverage']:.2%}, RMSE(valid)={score['depth_on_valid_predictions']['rmse_m']} m" for name, score in scores.items()))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("nyuv2", "middlebury"), required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--mask")
    parser.add_argument("--prediction-type", choices=("metric_depth", "relative_depth", "inverse_depth_like"), required=True)
    parser.add_argument("--unit", choices=("m", "mm", "arbitrary"), required=True)
    parser.add_argument("--alignment", choices=("none", "median", "inverse_scale_shift"), required=True)
    parser.add_argument("--paths-config", default="configs/paths.local.yaml")
    parser.add_argument("--manifest")
    parser.add_argument("--output-dir", required=True)
    evaluate_saved(parser.parse_args())


if __name__ == "__main__":
    main()
