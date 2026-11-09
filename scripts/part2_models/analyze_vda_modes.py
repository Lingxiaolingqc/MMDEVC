"""Verify mode exports and compare temporal diagnostics, latency and fixed-scale views."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from _common import ROOT, repo_path, checksum, write_json, write_csv
from _video_review import read_json, read_rows, verify_preview, contact_sheet
from _media import checked_comparison
from part3_improvement.run_fusion_suite import load_run
from p3_depth.evaluation.temporal import evaluate_relative_video


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clip", required=True, choices=("dev_tokyo_100", "course_rollercoaster_70"))
    a = p.parse_args()
    split = "dev" if a.clip == "dev_tokyo_100" else "test"
    cache = ROOT / "outputs/shared/video_cache" / a.clip
    parent = ROOT / "outputs/part2_models/vda_modes" / split / a.clip
    out = parent / "analysis"
    if out.exists():
        raise FileExistsError("Preserve previous analysis")
    cm = read_json(cache / "metadata.json")
    for f, key in (("frames.npy", "frames_sha256"), ("frames.csv", "index_sha256")):
        if checksum(cache / f) != cm[key]:
            raise ValueError("Cache identity")
    times = [float(r["timestamp_s"]) for r in read_rows(cache / "frames.csv")]
    rgb = np.load(cache / "frames.npy", mmap_mode="r", allow_pickle=False)
    flow = ROOT / "outputs/part3_improvement/flow" / split / a.clip
    fm = read_json(flow / "metadata.json")
    if fm["cache_metadata"] != cm:
        raise ValueError("Flow cache differs")
    for name, sha in fm["artifacts_sha256"].items():
        if checksum(flow / (name + ".npy")) != sha:
            raise ValueError("Flow source changed")
    backward = np.load(flow / "backward.npy", mmap_mode="r", allow_pickle=False)
    common = np.load(flow / "common_mask.npy", mmap_mode="r", allow_pickle=False)
    rows, preds, masks, limits, summaries, pair_rows = [], [], [], [], {}, []
    sources = {}
    metadata = []
    for mode in ("offline", "streaming"):
        run = parent / mode
        meta, pred, valid = load_run(run, cm)
        for name, sha in meta["artifacts_sha256"].items():
            if checksum(run / name) != sha:
                raise ValueError("Mode artifact changed: " + name)
        for name, sha in meta["implementation_sha256"].items():
            if checksum(ROOT / name) != sha:
                raise ValueError("Mode implementation changed since inference")
        span = list(map(float, np.quantile(pred[valid], [.01, .99])))
        verify_preview(cache, run, times, span)
        stat, pairs = evaluate_relative_video(pred, valid, backward, common, times, normalization_limits=span)
        timing = read_rows(run / "timing_repeats.csv")
        get = lambda key: np.array([float(r[key]) for r in timing])
        record = {
            "mode": mode, "frames": len(pred), "repeats": len(timing),
            "median_total_s": float(np.median(get("seconds"))),
            "min_total_s": float(np.min(get("seconds"))), "max_total_s": float(np.max(get("seconds"))),
            "median_amortized_ms": float(np.median(get("amortized_ms_per_output"))),
            "median_fps": float(np.median(get("fps"))),
            "median_first_api_return_ms": float(np.median(get("first_api_return_ms"))),
            "subsequent_median_ms": None, "subsequent_p95_ms": None,
            "max_allocated_mib": float(np.max(get("peak_allocated_mib"))),
            "max_reserved_mib": float(np.max(get("peak_reserved_mib"))),
            "T_own_span": stat["capped_normalized_l1_full_reference"],
            "raw_relative_l1": stat["temporal_l1_relative_on_valid"],
            "normalization_low": span[0], "normalization_high": span[1],
            "reference_pixels": stat["reference_pixels"], "pair_coverage": stat["pair_coverage"],
            "prefix_reset_max_abs": meta["prefix_reset_max_absolute_error"]
        }
        if mode == "streaming":
            latency = read_rows(run / "per_frame_latency.csv")
            if len(latency) != len(pred) * len(timing):
                raise ValueError("Incomplete latency trace")
            dt = np.array([float(r["seconds"])*1000 for r in latency if int(r["frame"]) > 0])
            record["subsequent_median_ms"] = float(np.median(dt))
            record["subsequent_p95_ms"] = float(np.quantile(dt, .95))
            if meta["prefix_reset_max_absolute_error"] > 1e-4:
                raise ValueError("Prefix state reset failed")
        rows.append(record)
        summaries[mode] = stat
        pair_rows.extend({"mode": mode, **r} for r in pairs)
        preds.append(pred); masks.append(valid); limits.append(span); metadata.append(meta)
        sources[(run / "metadata.json").relative_to(ROOT).as_posix()] = checksum(run / "metadata.json")
    for key in ("asset", "device_name", "precision", "input_size", "threads", "torch_version", "cuda_version"):
        if metadata[0]["model"][key] != metadata[1]["model"][key]:
            raise ValueError("Unmatched experiment setting: " + key)
    old = ROOT / "outputs/part2_models/vda_relative" / split / a.clip
    _, old_pred, _ = load_run(old, cm)
    difference = float(np.max(np.abs(old_pred - preds[0])))
    out.mkdir(); (out / "figures").mkdir()
    write_csv(out / "comparison.csv", rows)
    write_csv(out / "per_pair.csv", pair_rows)
    checked_comparison(cache, [parent / "offline", parent / "streaming"], parent / "visualizations")
    overview = np.linspace(0, len(rgb)-1, 6).round().astype(int).tolist()
    windows = ([list(range(0, 6)), list(range(28, 34)), list(range(60, 66)), list(range(94, 100))]
               if split == "dev" else
               [list(range(0, 6)), list(range(14, 20)), list(range(32, 38)), list(range(64, 70))])
    roi = [200, 60, 390, 350] if split == "dev" else [100, 60, 450, 260]
    contact_sheet(rgb, preds, masks, limits, ["offline", "streaming"], times,
                  overview, out / "figures/overview.png")
    for w in windows:
        contact_sheet(rgb, preds, masks, limits, ["offline", "streaming"], times,
                      w, out / f"figures/roi_{w[0]}_{w[-1]}.png", roi=roi)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    for mode in ("offline", "streaming"):
        points = [r for r in pair_rows if r["mode"] == mode]
        axes[0].plot([r["timestamp_s"] for r in points],
                     [r["capped_normalized_l1_full_reference"] for r in points], label=mode)
    axes[0].set(xlabel="Time (s)", ylabel="T (own fixed span)", title="Variation, not depth accuracy")
    axes[0].legend()
    for repeat in range(metadata[1]["repetitions"]):
        trace = [r for r in latency if int(r["repeat"]) == repeat]
        axes[1].plot([int(r["frame"]) for r in trace],
                     [float(r["seconds"])*1000 for r in trace], label=f"run {repeat+1}")
    axes[1].set(xlabel="Frame", ylabel="Synchronized API ms", title="Streaming: first and cached calls")
    axes[1].legend()
    fig.savefig(out / "figures/diagnostics.png", dpi=170); plt.close(fig)
    write_json(out / "quantitative_review.json", {
        "status": "quantitative_complete_visual_pending", "clip": a.clip, "split": split,
        "source_sha256": sources, "same_checkpoint_settings_verified": True,
        "all_saved_arrays_preview_timestamps_verified": True, "prior_offline_max_abs_difference": difference,
        "overview_frames": overview, "consecutive_windows": windows, "roi_xyxy": roi,
        "selection": "uniform overview, initialization, fixed motion windows and final frames; not selected to favor streaming",
        "summaries": summaries,
        "measurement_limits": [
            "Three repetitions of each short clip, sequential mode blocks; not independent videos or randomized thermal/load control.",
            "First API return uses already-resident inputs; offline returns all frames, streaming one. Not live input waiting time.",
            "Streaming is causal at inference; whole-clip display normalization is only offline review.",
            "T uses per-mode clip spans and a common flow mask; no video GT, no accuracy claim.",
            "Original offline arrays are preserved; new timing never substituted silently into earlier experiments."
        ],
        "artifacts_sha256": {p.relative_to(out).as_posix(): checksum(p) for p in out.rglob("*") if p.is_file()},
        "implementation_sha256": checksum(Path(__file__))
    })
    print(rows, flush=True)
    print("PASS mode comparison, raw metrics, repeated timing and visual evidence:", out, flush=True)


if __name__ == "__main__":
    main()

