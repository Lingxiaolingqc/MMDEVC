"""Inspect local Middlebury Q data. No model inference, downloads or GPU required."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PIL import Image, ImageDraw
import yaml

from p3_depth.data.middlebury import EXPECTED_SCENES, discover_samples, load_sample, write_manifest


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_preview(sample, destination: Path) -> None:
    """Create a labeled diagnostic, not a model-result visualization."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e5e5e5")
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), constrained_layout=True)
    axes[0, 0].imshow(sample.left_rgb)
    axes[0, 0].set_title("Left RGB")
    axes[0, 1].imshow(sample.right_rgb)
    axes[0, 1].set_title("Right RGB")
    disparity = np.ma.masked_where(~sample.valid_gt, sample.disparity_px)
    depth = np.ma.masked_where(~sample.valid_gt, sample.depth_m)
    fig.colorbar(axes[0, 2].imshow(disparity, cmap=cmap), ax=axes[0, 2], label="pixels")
    axes[0, 2].set_title("GT disparity (raw PFM values)")
    fig.colorbar(axes[1, 0].imshow(depth, cmap=cmap), ax=axes[1, 0], label="meters")
    axes[1, 0].set_title("GT-derived Z depth")
    axes[1, 1].imshow(sample.nonoccluded, cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title("Nonoccluded: mask == 255")
    overlay = sample.left_rgb.astype(np.float32)
    occluded = sample.occlusion_labels == 128
    unknown = sample.occlusion_labels == 0
    overlay[occluded] = overlay[occluded] * 0.4 + np.array([255, 100, 0]) * 0.6
    overlay[unknown] = [255, 0, 255]
    axes[1, 2].imshow(overlay.astype(np.uint8))
    axes[1, 2].set_title("Orange: occluded; magenta: unknown")
    for axis in axes.flat:
        axis.axis("off")
    fig.suptitle(f"{sample.paths.sample_id} | data inspection only; per-scene GT display ranges")
    fig.savefig(destination, dpi=120)
    plt.close(fig)


def make_contact_sheet(previews: list[Path], destination: Path) -> None:
    tile_w, tile_h, columns = 520, 285, 3
    rows = (len(previews) + columns - 1) // columns
    sheet = Image.new("RGB", (tile_w * columns, tile_h * rows), "white")
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(previews):
        x, y = (index % columns) * tile_w, (index // columns) * tile_h
        with Image.open(path) as image:
            thumbnail = image.convert("RGB")
            thumbnail.thumbnail((tile_w, tile_h - 20))
            sheet.paste(thumbnail, (x, y + 20))
        draw.text((x + 8, y + 4), path.stem, fill="black")
    sheet.save(destination)


def run(args: argparse.Namespace) -> Path:
    config_path = repo_path(args.paths_config)
    settings = {}
    if config_path.exists():
        settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(settings, dict):
            raise ValueError("Paths configuration must be a mapping")
    elif args.data_root is None:
        raise FileNotFoundError("Create configs/paths.local.yaml or pass --data-root")
    data_roots = settings.get("data_roots", {})
    if not isinstance(data_roots, dict):
        raise ValueError("data_roots must be a mapping")
    configured_root = args.data_root or data_roots.get("middlebury")
    if not configured_root:
        raise ValueError("Set data_roots.middlebury or pass --data-root")
    data_root = repo_path(configured_root)
    samples = discover_samples(data_root)
    actual = {sample.sample_id for sample in samples}
    expected = set(EXPECTED_SCENES)
    if actual != expected:
        raise ValueError(f"Expected the 15 trainingQ scenes: missing={sorted(expected-actual)}, "
                         f"unexpected={sorted(actual-expected)}")
    if args.write_manifest and not args.all:
        raise ValueError("Use --all with --write-manifest so every row is validated first")
    selected = samples if args.all else [s for s in samples if s.sample_id == args.sample]
    if not selected:
        raise ValueError(f"Unknown sample {args.sample}; choose one of {sorted(actual)}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = repo_path(args.output_dir) if args.output_dir else (
        repo_path(settings.get("output_root", "outputs")) / f"middlebury_inspection_{stamp}"
    )
    if output.exists():
        raise FileExistsError(f"Use a new output directory; will not overwrite {output}")
    output.mkdir(parents=True)
    summaries, provenance, previews = [], [], []
    for paths in selected:
        sample = load_sample(paths)
        d = sample.disparity_px[sample.valid_gt]
        z = sample.depth_m[sample.valid_gt]
        calib = sample.calibration
        row = {
            "sample_id": paths.sample_id, "width": calib.width, "height": calib.height,
            "gt_valid_fraction": float(sample.valid_gt.mean()),
            "nonoccluded_fraction": float(sample.nonoccluded.mean()),
            "occluded_fraction": float((sample.occlusion_labels == 128).mean()),
            "unknown_fraction": float((sample.occlusion_labels == 0).mean()),
            "disparity_min_px": float(d.min()), "disparity_max_px": float(d.max()),
            "depth_min_m": float(z.min()), "depth_median_m": float(np.median(z)),
            "depth_max_m": float(z.max()), "focal_px": calib.focal_px,
            "baseline_mm": calib.baseline_mm, "doffs_px": calib.doffs_px,
            "pfm_header_scale": sample.pfm_header_scale,
        }
        summaries.append(row)
        files = [paths.left, paths.right, paths.calibration, paths.disparity_gt, paths.nonocclusion_mask]
        provenance.append({"sample_id": paths.sample_id, "files": [
            {"path": path.relative_to(data_root).as_posix(), "sha256": checksum(path)} for path in files
        ]})
        if not args.no_preview:
            image_path = output / f"{paths.sample_id}.png"
            make_preview(sample, image_path)
            previews.append(image_path)
        if args.save_arrays:
            np.save(output / f"{paths.sample_id}_gt_depth_m.npy", sample.depth_m)
            np.save(output / f"{paths.sample_id}_gt_valid.npy", sample.valid_gt)
            np.save(output / f"{paths.sample_id}_nonoccluded.npy", sample.nonoccluded)
        print(f"PASS {paths.sample_id}: {calib.width}x{calib.height}, "
              f"GT={row['gt_valid_fraction']:.2%}, nonocc={row['nonoccluded_fraction']:.2%}, "
              f"Z={z.min():.3f}..{z.max():.3f} m")
    if args.write_manifest:
        write_manifest(samples, data_root, repo_path(args.manifest))
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)
    if len(previews) > 1:
        make_contact_sheet(previews, output / "contact_sheet.png")

    def git(*arguments: str) -> str | None:
        result = subprocess.run(["git", *arguments], cwd=ROOT, text=True, capture_output=True)
        return result.stdout.strip() if result.returncode == 0 else None

    manifest = repo_path(args.manifest)
    metadata = {
        "kind": "data_inspection_not_model_results", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": git("rev-parse", "HEAD"), "working_tree_status": git("status", "--porcelain"),
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ["numpy", "Pillow", "PyYAML", "matplotlib"]},
        "implementation_sha256": {path.relative_to(ROOT).as_posix(): checksum(path) for path in [
            Path(__file__).resolve(), ROOT / "src/p3_depth/data/middlebury.py"]},
        "manifest_sha256": checksum(manifest) if manifest.is_file() else None,
        "selection": [sample.sample_id for sample in selected], "sources": provenance,
        "pfm_policy": "header sign selects endianness; flip rows; preserve raw disparity values",
        "depth_formula": "f_px * baseline_mm / (1000 * (disparity_px + doffs_px))",
        "mask_policy": {"0": "unknown", "128": "occluded", "255": "nonoccluded"},
        "development_split": "not established by this inspection",
        "arguments": vars(args),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Inspected {len(selected)}/{len(samples)} training scenes. Output: {output}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", default="configs/paths.local.yaml")
    parser.add_argument("--data-root", help="Override the Middlebury data root")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--sample", default="Adirondack")
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--manifest", default="manifests/middlebury.csv")
    parser.add_argument("--output-dir", help="New directory; defaults to a unique outputs/ run")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--save-arrays", action="store_true", help="Save derived GT depth and boolean masks")
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
