"""Validate/extract the original labeled MAT and preserve the 795/654 standard split."""

import argparse
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from run_stereo_baseline import ROOT, checksum, read_yaml, repo_path, write_csv, write_json
from p3_depth.data.nyuv2 import NYU_CROP_YX, NYUv2Labeled, read_standard_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", default="configs/paths.local.yaml")
    parser.add_argument("--output-dir", required=True, help="New inspection output directory")
    args = parser.parse_args()
    data_root = repo_path(read_yaml(repo_path(args.paths_config))["data_roots"]["nyuv2"])
    labeled, split_file = data_root / "nyu_depth_v2_labeled.mat", data_root / "splits.mat"
    splits = read_standard_split(split_file)
    train_ids = set(splits["train"].tolist())
    output = repo_path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Will not overwrite {output}")
    output.mkdir(parents=True)
    processed = data_root / "processed"
    for folder in (processed / "rgb", processed / "depth"):
        folder.mkdir(parents=True, exist_ok=True)
    rows, stats, dev, dev_scenes, previews = [], [], [], set(), []
    with NYUv2Labeled(labeled) as dataset:
        for index in range(1, 1450):
            sample = dataset.load(index)
            if not sample.valid_gt.any():
                raise ValueError(f"Empty evaluation region: {sample.sample_id}")
            rgb_path = processed / "rgb" / f"{sample.sample_id}.png"
            depth_path = processed / "depth" / f"{sample.sample_id}.npy"
            # Resuming is safe only when existing decoded data are exactly equal.
            if rgb_path.exists():
                with Image.open(rgb_path) as existing:
                    if not np.array_equal(np.asarray(existing), sample.rgb):
                        raise ValueError(f"Existing RGB differs: {rgb_path}")
            else:
                Image.fromarray(sample.rgb).save(rgb_path)
            if depth_path.exists():
                previous = np.load(depth_path, allow_pickle=False)
                if previous.dtype != np.float32 or not np.array_equal(previous, sample.depth_m, equal_nan=True):
                    raise ValueError(f"Existing depth differs: {depth_path}")
            else:
                np.save(depth_path, sample.depth_m, allow_pickle=False)
            row = {"sample_id": sample.sample_id,
                   "rgb_path": rgb_path.relative_to(data_root).as_posix(),
                   "depth_gt_path": depth_path.relative_to(data_root).as_posix(),
                   "split": "train" if index in train_ids else "test",
                   "notes": f"original MATLAB index={index}; scene={sample.scene}; filled depths in meters"}
            rows.append(row)
            z = sample.depth_m[sample.valid_gt]
            stats.append({"sample_id": sample.sample_id, "split": row["split"], "scene": sample.scene,
                          "height": 480, "width": 640, "gt_eval_pixels": int(sample.valid_gt.sum()),
                          "depth_min_m": float(z.min()), "depth_median_m": float(np.median(z)), "depth_max_m": float(z.max())})
            if index in train_ids and len(dev) < 3 and sample.scene not in dev_scenes:
                dev.append(row)
                dev_scenes.add(sample.scene)
                previews.append(sample)
                np.save(output / f"{sample.sample_id}_eval_mask.npy", sample.valid_gt)
            if index % 100 == 0:
                print(f"Validated/extracted {index}/1449", flush=True)
    if len(dev) != 3:
        raise ValueError("Expected three training-side development scenes")
    write_csv(ROOT / "manifests/nyuv2.csv", rows)
    write_csv(ROOT / "manifests/nyuv2_dev.csv", dev)
    write_csv(output / "summary.csv", stats)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(12, 8), constrained_layout=True)
    for number, sample in enumerate(previews):
        axes[number, 0].imshow(sample.rgb)
        axes[number, 0].set_title(f"{sample.sample_id} | {sample.scene}")
        plot = axes[number, 1].imshow(sample.depth_m, vmin=0, vmax=10, cmap="viridis")
        axes[number, 1].set_title("Filled GT depth [0,10] m")
        fig.colorbar(plot, ax=axes[number, 1], label="m", shrink=.7)
        axes[number, 2].imshow(sample.valid_gt, vmin=0, vmax=1, cmap="gray")
        axes[number, 2].set_title("Fixed crop + valid GT range")
    for axis in axes.flat:
        axis.axis("off")
    fig.savefig(output / "development_samples.png", dpi=120)
    plt.close(fig)
    metadata = {"kind": "nyuv2_data_validation", "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "working_tree_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(),
                "counts": {"train": 795, "test": 654}, "development_ids": [r["sample_id"] for r in dev],
                "development_selection": "first three distinct scenes in ascending MATLAB index, training side only",
                "labeled_source": "https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat",
                "labeled_sha256": checksum(labeled), "labeled_bytes": labeled.stat().st_size,
                "split_source": "https://github.com/ankurhanda/nyuv2-meta-data/blob/master/splits.mat",
                "split_sha256": checksum(split_file), "matlab_indexing": "1-based; ID preserves original frame index",
                "axis_policy": "images N,C,W,H -> H,W,C; depths N,W,H -> H,W; no rotation or mirroring",
                "depth_field": "depths (inpainted); original float32 meters, no PNG quantization",
                "crop_y0_y1_x0_x1": list(NYU_CROP_YX), "gt_interval_m_inclusive": [0.001, 10.0],
                "manifest_sha256": checksum(ROOT / "manifests/nyuv2.csv"),
                "dev_manifest_sha256": checksum(ROOT / "manifests/nyuv2_dev.csv"),
                "evaluation_note": "DA V2 metric formulas; current upstream has no NYUv2 loader; crop supplemented from UniDepth official NYUv2 eval_mask"}
    write_json(output / "metadata.json", metadata)
    write_json(data_root / "extraction_metadata.json", metadata)
    print(f"PASS: 1449 pairs; 795 train / 654 test; dev={[r['sample_id'] for r in dev]}")


if __name__ == "__main__":
    main()
