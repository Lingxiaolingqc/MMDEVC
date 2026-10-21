"""Run pinned official UniDepthV2 ViT-S on one training-side NYUv2 development image."""

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from time import perf_counter

import numpy as np
from PIL import Image

from run_stereo_baseline import ROOT, checksum, read_yaml, repo_path, write_csv, write_json
from evaluate_depth import evaluate_saved

UPSTREAM_COMMIT = "8d8cfe4c7ee15297099983607febf0d4f32eb3d6"
WEIGHT_REVISION = "038c238f06c87b6c2f5b3749fd51fbf442b1f218"
WEIGHT_SHA256 = "93705cb3295dd7476b44911b8a55f5215bf74e8d5eccd27cecdb1b338270a648"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", default="nyuv2_0003")
    parser.add_argument("--paths-config", default="configs/paths.local.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resolution-level", type=int, choices=range(10), default=0)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    output = repo_path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Will not overwrite {output}")
    settings = read_yaml(repo_path(args.paths_config))
    source = repo_path(settings["third_party_root"]) / "UniDepth"
    weights = repo_path(settings["checkpoint_root"]) / "unidepth-v2-vits14"
    upstream_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if upstream_sha != UPSTREAM_COMMIT or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip():
        raise ValueError("Use the pinned, clean official UniDepth checkout")
    weight_path = weights / "model.safetensors"
    if checksum(weight_path) != WEIGHT_SHA256:
        raise ValueError("Checkpoint SHA256 mismatch")
    config_path = weights / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    with (ROOT / "manifests/nyuv2_dev.csv").open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["sample_id"] == args.sample]
    if len(rows) != 1 or rows[0]["split"] != "train":
        raise ValueError("This smoke runner accepts only the declared training-side development samples")
    row = rows[0]
    data_root = repo_path(settings["data_roots"]["nyuv2"])
    rgb_path = (data_root / row["rgb_path"]).resolve()
    gt_path = (data_root / row["depth_gt_path"]).resolve()
    if not all(p.is_relative_to(data_root) for p in (rgb_path, gt_path)):
        raise ValueError("Manifest paths must stay inside the data root")
    with Image.open(rgb_path) as handle:
        rgb = np.array(handle.convert("RGB"))
    if rgb.shape != (480, 640, 3):
        raise ValueError("Expected the original NYUv2 RGB dimensions")

    # Inference is offline; no experiment tracking or implicit checkpoint downloads.
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    sys.path.insert(0, str(source))
    import torch
    from safetensors.torch import load_file
    from unidepth.models import UniDepthV2
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = "cpu" if device == "auto" else device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but this PyTorch/environment cannot use it")
    torch.set_num_threads(args.threads)
    torch.manual_seed(13)
    print(f"Loading official ViT-S on {device}; strict checkpoint load", flush=True)
    model = UniDepthV2(config)
    model.load_state_dict(load_file(str(weight_path), device="cpu"), strict=True)
    model = model.eval().to(device)
    model.resolution_level = args.resolution_level
    tensor = torch.from_numpy(rgb).permute(2, 0, 1)
    input_shapes = []
    hook = model.pixel_encoder.register_forward_pre_hook(lambda _, inputs: input_shapes.append(list(inputs[0].shape)))
    print("Warm-up: one official infer call", flush=True)
    with torch.inference_mode():
        warm = model.infer(tensor)
        del warm
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        prediction = model.infer(tensor)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = perf_counter() - start
    hook.remove()
    depth = prediction["depth"][0, 0].float().cpu().numpy().copy()
    if depth.shape != rgb.shape[:2]:
        raise ValueError("Official infer did not restore original spatial dimensions")
    valid = np.isfinite(depth) & (depth > 0)
    depth[~valid] = np.nan
    for folder in (output / "predictions", output / "masks", output / "visualizations"):
        folder.mkdir(parents=True, exist_ok=True)
    depth_path, mask_path = output / "predictions" / f"{args.sample}.npy", output / "masks" / f"{args.sample}.npy"
    np.save(depth_path, depth)
    np.save(mask_path, valid)
    write_csv(output / "predictions.csv", [{"sample_id": args.sample, "clip_id": "", "frame_index": "", "timestamp_s": "",
        "prediction_path": depth_path.relative_to(output).as_posix(), "mask_path": mask_path.relative_to(output).as_posix()}])
    write_json(output / "config_resolved.json", {"arguments": vars(args), "upstream_model_config": config})
    evaluation_args = argparse.Namespace(dataset="nyuv2", sample=args.sample, prediction=str(depth_path), mask=str(mask_path),
        prediction_type="metric_depth", unit="m", alignment="none", paths_config=args.paths_config,
        manifest=None, output_dir=str(output / "evaluation"))
    evaluate_saved(evaluation_args)
    gt = np.load(gt_path, allow_pickle=False)
    from p3_depth.data.nyuv2 import evaluation_mask
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title(f"{args.sample} | RGB")
    for axis, array, title, top in zip(axes[1:], [gt, depth, np.where(evaluation_mask(gt) & valid, np.abs(depth-gt), np.nan)],
            ["GT Z depth [m]", "UniDepthV2 Z depth [m]", "Abs error in eval crop [m]"], [10, 10, 1]):
        plot = axis.imshow(array, vmin=0, vmax=top, cmap="viridis")
        axis.set_title(title)
        fig.colorbar(plot, ax=axis, shrink=.7)
    for axis in axes:
        axis.axis("off")
    fig.savefig(output / "visualizations" / f"{args.sample}.png", dpi=120)
    plt.close(fig)
    packages = ["torch", "torchvision", "numpy", "Pillow", "h5py", "scipy", "matplotlib", "opencv-python",
                "timm", "einops", "wandb", "safetensors", "huggingface-hub"]
    metadata = {"method": "UniDepthV2 ViT-S", "sample_id": args.sample, "split": "train_development",
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "working_tree_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(),
        "upstream_commit": upstream_sha, "checkpoint_repository": "lpiccinelli/unidepth-v2-vits14",
        "checkpoint_revision": WEIGHT_REVISION, "weight_sha256": WEIGHT_SHA256, "config_sha256": checksum(config_path),
        "manifest_sha256": checksum(ROOT / "manifests/nyuv2_dev.csv"), "input_rgb_sha256": checksum(rgb_path),
        "gt_sha256": checksum(gt_path), "prediction_sha256": checksum(depth_path), "mask_sha256": checksum(mask_path),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "packages": {name: importlib.metadata.version(name) for name in packages}},
        "device": device, "device_name": torch.cuda.get_device_name() if device == "cuda" else platform.processor(),
        "precision": "official CUDA autocast float16" if device == "cuda" else "float32; CUDA autocast disabled by PyTorch",
        "threads": torch.get_num_threads(), "parameters": sum(p.numel() for p in model.parameters()),
        "input_shape_hwc": list(rgb.shape), "encoder_input_shapes_nchw": input_shapes, "output_shape_hw": list(depth.shape),
        "resolution_level": args.resolution_level, "camera_input": "none; intrinsics predicted by the model",
        "predicted_intrinsics": prediction["intrinsics"].float().cpu().tolist(),
        "prediction_type": "metric_depth", "unit": "m", "depth_convention": "camera Z, not radial distance",
        "alignment": "none", "gt_used_by_model": False, "invalid_policy": "nonpositive/nonfinite -> NaN; separate bool mask",
        "preprocessing": "official infer: ImageNet normalization, aspect padding, bilinear resize; official postprocess restores original shape",
        "warmup_calls": 1, "timed_calls": 1, "seconds_official_infer": elapsed,
        "timing_scope": "wall time of official infer including internal preprocessing and device transfer; excludes model load, file IO and evaluation",
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else None,
        "reproduce": f"python scripts/run_unidepth_sample.py --sample {args.sample} --device {device} --resolution-level {args.resolution_level} --threads {args.threads} --output-dir outputs/unidepth_reproduce_NEW",
        "display_ranges_m": {"gt_and_prediction": [0, 10], "absolute_error": [0, 1]}}
    write_json(output / "metadata.json", metadata)
    (output / "run.log").write_text(f"PASS {args.sample}; direct metric; device={device}; infer={elapsed:.6f}s; no GT alignment\n", encoding="utf-8")
    print((output / "run.log").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
