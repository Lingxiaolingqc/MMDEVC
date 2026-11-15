"""Package the new frozen model and required demonstrations independently."""
import json
from pathlib import Path

from depth_project.metric_training import write_json
from depth_project.submission import build_submission


def main():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/metric-research-final.json").read_text())
    paths = ["configs/metric-research-final.json", "results/metric-research/final-test.json",
             "results/metric-research/scene-fold-selection.json", "results/metric-research/seed-selection.json",
             "results/metric-research/refit-decision.json", "results/metric-research/isolated-benchmark.json",
             "results/metric-research/normal-entry-development-replay.json", "results/metric-research/independent-metric-check.json",
             "results/metric-research/training-cost.json",
             "results/metric-research/figure-selection.json", "results/metric-research/learning-curves.png",
             "results/metric-research/curves-analysis.json", "results/metric-research/lora-pilots.json"]
    for specification in [config["pre_refit"], config["selected"]]:
        components = [specification["lora"], specification["refiner"]] if specification.get("type") == "lora-feature-mean" else [specification]
        for component in components:
            paths.extend(member["checkpoint"] for member in component["members"])
    paths.extend(str(path.relative_to(root)).replace("\\", "/")
                 for path in sorted((root / "results/metric-research/figures").glob("*.png")))
    paths.extend(f"results/metric-research/{name}-test-per-image.csv"
                 for name in ["baseline", "previous_refiner", "pre_refit", "selected"])
    files = {path: root / path for path in sorted(set(paths))}
    inventory = build_submission(root / "runs/final", root / "outputs/metric-research/depth_results.zip",
                                 allow_short=True, report_path=root / "report/metric-research-report.pdf",
                                 supplemental_files=files)
    write_json(root / "results/metric-research/archive-inventory.json", inventory)
    print(json.dumps({"files": len(inventory["files"]) + 1, "frames": inventory["processed_frames"],
                      "warning": inventory["warning"]}, indent=2))


if __name__ == "__main__":
    main()
