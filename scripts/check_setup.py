"""Check the repository scaffold using only the Python standard library."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md", ".gitignore", "pyproject.toml", "CONTRIBUTING.md",
    "docs/weekly_plan.md", "docs/week1_plan.md", "docs/data_format.md",
    "docs/evaluation_protocol.md", "docs/open_questions.md", "docs/task_board.md",
    "configs/project.yaml", "configs/paths.example.yaml", "configs/models.example.yaml",
    "configs/ablations/matrix.yaml", "environments/model_sources.yaml",
    "src/p3_depth/__init__.py",
)
HEADERS = {
    "manifests/middlebury.csv": "sample_id,left_path,right_path,disparity_gt_path,calibration_path,split,notes",
    "manifests/nyuv2.csv": "sample_id,rgb_path,depth_gt_path,split,notes",
    "manifests/videos.csv": "clip_id,video_path,scene_group,split,scene_type,fps,height,width,duration_s,notes",
    "manifests/frames.example.csv": "clip_id,frame_index,timestamp_s,rgb_path,height,width",
    "results/run_index.csv": "run_id,method,dataset,split,code_commit,config_path,summary_path,artifact_location,reviewer,notes",
}


def main() -> int:
    errors: list[str] = []
    if sys.version_info < (3, 10):
        errors.append("Python 3.10+ is required for the scaffold.")
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append(f"Missing file: {relative}")

    for relative, expected in HEADERS.items():
        try:
            with (ROOT / relative).open(encoding="utf-8-sig", newline="") as handle:
                rows = csv.reader(handle)
                if next(rows, None) != expected.split(","):
                    errors.append(f"Unexpected CSV header: {relative}")
                for line_number, row in enumerate(rows, start=2):
                    if len(row) != len(expected.split(",")):
                        errors.append(f"Wrong column count: {relative}:{line_number}")
        except (OSError, UnicodeError, csv.Error) as exc:
            errors.append(f"Cannot read {relative}: {exc}")

    try:
        metadata = json.loads((ROOT / "configs/metadata.example.json").read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            errors.append("Metadata template must be a JSON object.")
        else:
            for field in ("schema_version", "status", "run_id", "code_commit", "checkpoint",
                          "dataset", "prediction_type", "unit", "alignment", "environment"):
                if field not in metadata:
                    errors.append(f"Missing metadata template field: {field}")
            if metadata.get("status") != "template_not_a_result":
                errors.append("The metadata example must remain explicitly labeled as a template.")
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"Cannot read metadata template: {exc}")

    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print("PASS: repository scaffold, CSV structure and metadata template.")
    if not (ROOT / "configs/paths.local.yaml").is_file():
        print("NEXT: copy configs/paths.example.yaml to paths.local.yaml and edit local paths.")
    print("Not checked: dataset contents, model environments, predictions, metrics or GPU readiness.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
