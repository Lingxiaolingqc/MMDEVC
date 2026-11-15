"""Independently recompute course metrics from every frozen test packet."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from depth_project.metric_training import write_json
from depth_project.records import sha256_file


def save_check_result(run_directory, result, output=None):
    destination = Path(output) if output is not None else Path(run_directory) / "independent-metric-check.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, result)
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/metric-research-final"))
    parser.add_argument("--previous-run-dir", type=Path, help="Also compare against preserved historical float packets")
    parser.add_argument("--output", type=Path, help="Check output, defaulting to the selected run directory")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    folder = root / args.run_dir
    summary = json.loads((folder / "summary.json").read_text())
    names = ["baseline", "previous_refiner", "pre_refit", "selected"]
    scores = {name: [] for name in names}
    old_differences = {"baseline": 0.0, "previous_refiner": 0.0}
    identities = summary["test_ids"]
    if len(identities) != 654 or len(set(identities)) != 654:
        raise ValueError("Independent check requires every unique official test image")
    with h5py.File(root / "data/nyu/nyu_depth_v2_labeled.mat") as dataset:
        for index in identities:
            truth = dataset["depths"][index].T.astype(np.float64)[45:471, 41:601]
            mask = np.isfinite(truth) & (truth > 0.001) & (truth < 10)
            target = truth[mask]
            path = folder / f"prediction_{index:04d}.npz"
            if sha256_file(path) != summary["prediction_files_sha256"][str(index)]:
                raise ValueError("Frozen packet content identity differs")
            with np.load(path, allow_pickle=False) as packet:
                for name in names:
                    values = packet[f"{name}_m"]
                    if values.shape != (480, 640) or not np.isfinite(values).all():
                        raise ValueError("Invalid metric prediction dimensions or values")
                    prediction = np.clip(values.astype(np.float64), 0.001, 10)[45:471, 41:601][mask]
                    error = prediction - target
                    scores[name].append([float(np.mean(np.abs(error) / target)),
                                         float(np.sqrt(np.mean(error ** 2))),
                                         float(np.mean(np.maximum(prediction / target, target / prediction) < 1.25))])
                if args.previous_run_dir:
                    with np.load(root / args.previous_run_dir / f"prediction_{index:04d}.npz", allow_pickle=False) as old:
                        old_differences["baseline"] = max(old_differences["baseline"], float(np.max(np.abs(packet["baseline_m"] - old["baseline_m"]))))
                        old_differences["previous_refiner"] = max(old_differences["previous_refiner"], float(np.max(np.abs(packet["previous_refiner_m"] - old["depth_m"]))))
    metrics = {name: dict(zip(["abs_rel", "rmse", "delta1"], np.mean(values, axis=0).tolist())) for name, values in scores.items()}
    maximum_delta = max(abs(value - summary["metrics"][name][key]) for name, values in metrics.items() for key, value in values.items())
    if maximum_delta > 1e-12:
        raise ValueError("Independent metric arithmetic differs from the evaluator")
    result = {"image_count": 654, "metrics": metrics, "maximum_metric_difference": maximum_delta,
              "historical_reference_packets_checked": 654 if args.previous_run_dir else 0,
              "maximum_previous_packet_difference_m": old_differences if args.previous_run_dir else None,
              "configuration_sha256": summary["configuration_sha256"],
              "scope": "Independent NumPy crop/mask/clip/formulas, equal-image aggregation, all packet SHA checks, preserved reference reproduction"}
    save_check_result(folder, result, root / args.output if args.output is not None else None)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
