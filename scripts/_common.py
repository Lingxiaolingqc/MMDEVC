"""Shared repository paths, file IO and manifest helpers for command-line scripts."""
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

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
    import yaml

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
    from p3_depth.data.middlebury import MANIFEST_FIELDS, SamplePaths

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
