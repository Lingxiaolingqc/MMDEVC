"""Matched repeated offline/streaming measurements and raw exports; no training."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import gc
import subprocess
from time import perf_counter
import numpy as np
from _common import ROOT, repo_path, checksum, read_yaml, write_json, write_csv
from _video_review import read_json
from _media import video_preview
from p3_depth.models.video import VideoModel
from p3_depth.models.streaming import StreamingVideoModel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("offline", "streaming"), required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--input-size", type=int, default=364)
    p.add_argument("--output-dir")
    a = p.parse_args()
    if a.repeats < 3:
        raise ValueError("Use at least three timing repetitions")
    import torch
    cache = repo_path(a.cache)
    cm = read_json(cache / "metadata.json")
    for name, key in (("frames.npy", "frames_sha256"), ("frames.csv", "index_sha256")):
        if checksum(cache / name) != cm[key]:
            raise ValueError("Cache changed")
    # Resident RGB cache: decode and disk IO are outside all measurements.
    frames = np.array(np.load(cache / "frames.npy", allow_pickle=False), copy=True)
    out = repo_path(a.output_dir) if a.output_dir else ROOT / "outputs/part2_models/vda_modes" / cm["split"] / cm["clip_id"] / a.mode
    if out.exists():
        raise FileExistsError("Preserve existing mode experiment")
    settings = read_yaml(ROOT / "configs/paths.local.yaml")
    t = perf_counter()
    if a.mode == "offline":
        model = VideoModel(ROOT, settings, "cuda", "fp16", a.input_size, 8, name="vda_relative")
        model.metadata["model"] = "vda_relative_offline"
    else:
        model = StreamingVideoModel(ROOT, settings, input_size=a.input_size)
    torch.cuda.synchronize()
    model_load_seconds = perf_counter() - t
    if a.mode == "offline":
        warm = model(frames[:32], cm["requested_fps"])
    else:
        warm = model(frames[0])
        warm = model(frames[1])
        model.reset()
    del warm
    timing, latency = [], []
    first, repeat_max_diff = None, []
    for repetition in range(a.repeats):
        if a.mode == "streaming":
            model.reset()
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        if a.mode == "offline":
            pred = model(frames, cm["requested_fps"])
            torch.cuda.synchronize()
            first_return = perf_counter() - start
        else:
            values = []
            for i, frame in enumerate(frames):
                torch.cuda.synchronize()
                t = perf_counter()
                value = model(frame)
                torch.cuda.synchronize()
                dt = perf_counter() - t
                latency.append({"repeat": repetition, "frame": i, "seconds": dt})
                values.append(value)
                if i == 0:
                    first_return = perf_counter() - start
            pred = np.stack(values)
        torch.cuda.synchronize()
        elapsed = perf_counter() - start
        allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
        if pred.shape != frames.shape[:3] or not np.isfinite(pred).all():
            raise ValueError("Incomplete or nonfinite mode prediction")
        if first is None:
            first = pred.copy()
        difference = float(np.max(np.abs(pred - first)))
        repeat_max_diff.append(difference)
        timing.append({"repeat": repetition, "mode": a.mode, "frames": len(pred),
                       "seconds": elapsed, "fps": len(pred) / elapsed,
                       "amortized_ms_per_output": 1000 * elapsed / len(pred),
                       "first_api_return_ms": 1000 * first_return,
                       "peak_allocated_mib": allocated / 1024**2,
                       "peak_reserved_mib": reserved / 1024**2,
                       "max_abs_difference_from_saved_repeat": difference})
        print(f"PASS {a.mode} repeat {repetition+1}: {elapsed:.3f}s; peak {allocated/1024**2:.1f} MiB", flush=True)
        del pred
    # Replay a prefix after a full clip: detects cache leakage/reset errors.
    prefix_check = None
    if a.mode == "streaming":
        model.reset()
        prefix = np.stack([model(frames[i]) for i in range(3)])
        prefix_check = float(np.max(np.abs(prefix - first[:3])))
        if prefix_check > 1e-4 or max(repeat_max_diff) > 1e-4:
            raise ValueError("Streaming reset/repetition consistency failed")
    out.mkdir(parents=True)
    np.save(out / "depths.npy", first)
    np.save(out / "valid.npy", np.isfinite(first))
    write_csv(out / "timing_repeats.csv", timing)
    if latency:
        write_csv(out / "per_frame_latency.csv", latency)
    info = {
        "status": "complete", "model": model.metadata, "cache_metadata": cm,
        "prediction_type": "inverse_depth_like", "unit": "arbitrary", "alignment": "none", "gt_used": False,
        "frames": len(first), "depths_sha256": checksum(out / "depths.npy"),
        "valid_sha256": checksum(out / "valid.npy"), "saved_repeat": 0,
        "inference_seconds": timing[0]["seconds"], "repetitions": a.repeats,
        "load_seconds_diagnostic": model_load_seconds,
        "timing_scope": "resident uint8 RGB through official preprocessing/inference/CPU float32 return; synchronized CUDA; excludes load/decode/disk/render",
        "first_return_definition": "all input frames already resident; offline official API returns full clip, streaming API returns first frame. Not intrinsic lookahead or live capture waiting time.",
        "warmup": "offline one 32-frame window; streaming first two calls followed by complete temporal-state reset; no weights updated",
        "repetition_policy": "same loaded model, streaming state reset and CUDA allocator cache cleared before every measured sequence",
        "prefix_reset_max_absolute_error": prefix_check,
        "repeat_max_absolute_errors": repeat_max_diff,
        "visualization_note": "full-clip fixed normalization is an offline review display, not an online color-scale algorithm",
        "code_commit": subprocess.check_output(["git","rev-parse","HEAD"], cwd=ROOT, text=True).strip(),
        "implementation_sha256": {n: checksum(ROOT / n) for n in (
            "scripts/part2_models/run_vda_mode.py", "src/p3_depth/models/streaming.py",
            "src/p3_depth/models/video.py", "src/p3_depth/models/runtime.py")},
        "artifacts_sha256": {x.name: checksum(x) for x in out.iterdir() if x.is_file()}
    }
    write_json(out / "metadata.json", info)
    video_preview(cache, out)
    print(f"PASS {a.mode}: raw arrays, repeated timing, and timestamped video saved to {out}", flush=True)


if __name__ == "__main__":
    main()

