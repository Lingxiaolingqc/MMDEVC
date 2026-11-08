"""Analyze verified confidence ablations without rerunning fusion or model inference.

Writes a new analysis directory. Visual-review conclusions are supplied separately
after the generated, provenance-linked contact sheets have actually been inspected.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
import numpy as np
import av
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import ROOT, repo_path, checksum, write_json, write_csv
from part3_improvement.run_supplementary_ablations import load_sources, read_json
from p3_depth.temporal.flow import sample_previous

T_KEY = "capped_normalized_l1_full_reference"
IDS = ["E0", "E3_recursive", "E4_recursive", "E4_no_photo",
       "E4_no_flow", "E4_no_depth", "E4_no_edge"]
LABELS = ["DA", "E3", "Full E4", "No photo", "No flow", "No depth", "No edge"]


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def close(actual, expected, label):
    if not np.isclose(actual, float(expected), rtol=2e-9, atol=2e-11):
        raise ValueError(f"{label} does not recompute: {actual} != {expected}")


def load_verified(run):
    meta, receipt = read_json(run / "metadata.json"), read_json(run / "verification.json")
    if meta["experiment"] != "confidence" or meta["status"] != "complete":
        raise ValueError("Require a completed confidence ablation")
    if receipt["status"] != "verified" or receipt["supplement_metadata_sha256"] != checksum(run / "metadata.json"):
        raise ValueError("Matching verification receipt required")
    verifier = ROOT / "scripts/part3_improvement/verify_supplementary_ablations.py"
    if receipt["verifier_sha256"] != checksum(verifier):
        raise ValueError("The verifier differs from the recorded verification")
    for rel, digest in meta["artifacts_sha256"].items():
        if checksum(run / rel) != digest:
            raise ValueError("Modified supplementary artifact: " + rel)
    for rel, digest in meta["implementation_sha256"].items():
        if checksum(ROOT / rel) != digest:
            raise ValueError("Supplementary implementation changed: " + rel)
    source = load_sources(*(repo_path(meta["sources"][k]) for k in
                            ("cache", "baseline", "flow", "reference")))
    if (meta["source_suite_sha256"] != checksum(source["reference"] / "metadata.json") or
            meta["source_verification_sha256"] != checksum(source["reference"] / "verification.json") or
            meta["parameters"] != source["sm"]["config"]["parameters"] or
            meta["normalization_limits"] != source["limits"]):
        raise ValueError("Supplementary and original protocols differ")
    rows = read_rows(run / "comparison.csv")
    if [r["id"] for r in rows] != IDS:
        raise ValueError("Require the seven ordered confidence controls")
    return meta, source, rows


def summarize(run, source, rows):
    """Recompute pooled T, raw L1, coverage, weights and departure from arrays."""
    common = source["fields"]["common_mask"]
    span = source["limits"][1] - source["limits"][0]
    summaries, pair_tables, arrays = [], {}, {}
    full = next(r for r in rows if r["id"] == "E4_recursive")
    for row in rows:
        name = row["id"]
        folder = (source["reference"] / row["variant"]
                  if row["reused_original"] == "True" else run / name)
        pred, valid, weight = (np.load(folder / (f + ".npy"), mmap_mode="r", allow_pickle=False)
                               for f in ("depths", "valid", "history_weight"))
        if pred.shape != source["base"].shape or not np.array_equal(valid, np.isfinite(pred)):
            raise ValueError("Invalid saved prediction shape/mask")
        saved = read_rows(folder / "per_pair.csv")
        if len(saved) != len(pred) - 1:
            raise ValueError("Incomplete pair table")
        table = []
        for t in range(1, len(pred)):
            history, history_valid = sample_previous(
                pred[t - 1], source["fields"]["backward"][t - 1], valid[t - 1])
            region = common[t - 1]
            chosen = region & valid[t] & history_valid
            n, k = int(region.sum()), int(chosen.sum())
            err = np.abs(pred[t][chosen].astype(np.float64) - history[chosen].astype(np.float64))
            score = float((np.minimum(err / span, 1).sum() + n - k) / n)
            current = {"id": name, "frame": t, "timestamp_s": source["times"][t],
                       "reference_pixels": n, "valid_pairs": k, "T": score,
                       "raw_L1": float(err.mean())}
            if int(saved[t - 1]["frame"]) != t:
                raise ValueError("Pair frame index changed")
            for key, expected in [(score, saved[t - 1][T_KEY]),
                                  (current["raw_L1"], saved[t - 1]["temporal_l1_relative_on_valid"]),
                                  (k / n, saved[t - 1]["pair_coverage"])]:
                close(key, expected, name + " pair")
            close(source["times"][t], saved[t - 1]["timestamp_s"], "pair timestamp")
            table.append(current)
        n = sum(x["reference_pixels"] for x in table)
        k = sum(x["valid_pairs"] for x in table)
        t_mean = sum(x["T"] * x["reference_pixels"] for x in table) / n
        raw_l1 = sum(x["raw_L1"] * x["valid_pairs"] for x in table) / k
        selected = common & valid[1:]
        w = weight[1:][selected].astype(np.float64)
        valid_both = valid & source["valid"]
        departure = float(np.abs(pred[valid_both].astype(np.float64) -
                                 source["base"][valid_both]) .mean() / span)
        for actual, expected, label in [
            (t_mean, row[T_KEY], "T"), (raw_l1, row["temporal_l1_relative"], "raw L1"),
            (k / n, row["pair_coverage"], "coverage"),
            (float(w.mean()), row["mean_history_weight_on_reference"], "history weight"),
            (departure, row["mean_departure_from_DA_normalized"], "departure")]:
            close(actual, expected, name + " " + label)
        item = {"id": name, "T": t_mean, "T_reduction_vs_full_E4_pct":
                100 * (1 - t_mean / float(full[T_KEY])),
                "departure_from_DA": departure,
                "departure_increase_vs_full_E4_pct":
                100 * (departure / float(full["mean_departure_from_DA_normalized"]) - 1),
                "mean_history_weight": float(w.mean()), "pair_coverage": k / n,
                "reference_pixels": n}
        summaries.append(item)
        pair_tables[name] = table
        arrays[name] = pred
        print("PASS raw diagnostics " + name, flush=True)
    full_pairs = pair_tables["E4_recursive"]
    for item in summaries:
        if item["id"].startswith("E4_no_"):
            delta = np.array([a["T"] - b["T"] for a, b in
                              zip(full_pairs, pair_tables[item["id"]])])
            item["lower_T_pairs"] = int((delta > 1e-12).sum())
            item["equal_T_pairs"] = int((np.abs(delta) <= 1e-12).sum())
            item["higher_T_pairs"] = int((delta < -1e-12).sum())
            item["largest_abs_T_difference_frame"] = int(np.argmax(np.abs(delta))) + 1
            # Determine another window from raw full-image differences, not rendered colors.
            diff = [float(np.abs(arrays[item["id"]][t].astype(np.float64) -
                                 arrays["E4_recursive"][t]).mean() / span)
                    for t in range(len(source["times"]))]
            item["largest_prediction_difference_frame"] = int(np.argmax(diff))
            item["mean_abs_difference_from_full_E4"] = float(np.mean(diff))
    return summaries, pair_tables


def plot_summary(out, summaries, pairs):
    colors = ["#808080", "#8f63a9", "#168c96", "#e89b39", "#6b9e42", "#c25c70", "#547ab7"]
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    x = np.arange(7)
    axs[0, 0].bar(x, [s["T"] for s in summaries], color=colors)
    axs[0, 0].set(title="Temporal variation T (fixed DA span)", ylabel="Pooled normalized change")
    axs[0, 1].bar(x, [s["departure_from_DA"] for s in summaries], color=colors)
    axs[0, 1].set(title="Departure from current DA prediction", ylabel="Mean normalized absolute difference")
    axs[1, 0].bar(x, [s["mean_history_weight"] for s in summaries], color=colors)
    axs[1, 0].set(title="History weight on common reference", ylabel="Mean weight", ylim=(0, .76))
    axs[1, 1].boxplot([[p["T"] for p in pairs[s["id"]]] for s in summaries],
                      tick_labels=LABELS, showfliers=True)
    axs[1, 1].set(title="Distribution over adjacent frame pairs", ylabel="Per-pair T")
    for ax in axs.flat:
        ax.grid(axis="y", alpha=.2)
    for ax in list(axs.flat)[:3]:
        ax.set_xticks(x, LABELS, rotation=25)
    axs[1, 1].tick_params(axis="x", labelrotation=25)
    fig.suptitle("Confidence leave-one-out diagnostics | not depth accuracy\n"
                 "Frame pairs are correlated; plots are descriptive, not significance tests.", fontsize=12)
    for suffix in ("png", "pdf"):
        fig.savefig(out / ("confidence_diagnostics." + suffix), dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4), layout="constrained")
    for i, name in enumerate(IDS[2:], 2):
        ax.plot([p["timestamp_s"] for p in pairs[name]], [p["T"] for p in pairs[name]],
                color=colors[i], label=LABELS[i], linewidth=1.3)
    ax.set(xlabel="Original timestamp (s)", ylabel="Per-pair T", title="Full E4 and confidence controls")
    ax.legend(ncol=3); ax.grid(alpha=.2)
    fig.savefig(out / "temporal_curve.png", dpi=180)
    plt.close(fig)


def make_sheet(decoded, indices, labels, width, height, times, target, roi=None):
    font = ImageFont.truetype("arial.ttf", 16)
    x0, y0, x1, y1 = roi or (0, 0, width, height)
    scale = 1.0 if roi else .5
    pw, ph = round((x1 - x0) * scale), round((y1 - y0) * scale)
    sheet = Image.new("RGB", (4 * pw, len(indices) * (ph + 28)), "white")
    d = ImageDraw.Draw(sheet)
    for row, frame in enumerate(indices):
        for col, label in enumerate(labels):
            panel = decoded[frame].crop((col * width + x0, 40 + y0,
                                         col * width + x1, 40 + y1))
            panel = panel.resize((pw, ph), Image.Resampling.LANCZOS)
            y = row * (ph + 28)
            sheet.paste(panel, (col * pw, y + 28))
            d.text((col * pw + 4, y + 4), f"{label} | f{frame} {times[frame]:.2f}s",
                   fill="black", font=font)
    sheet.save(target)


def visual_evidence(run, out, source, summaries, roi):
    width, height = source["base"].shape[2], source["base"].shape[1]
    if roi is not None and not (0 <= roi[0] < roi[2] <= width and 0 <= roi[1] < roi[3] <= height):
        raise ValueError("ROI must fit the original image grid")
    n = len(source["times"])
    overview = sorted(set(np.linspace(0, n - 1, 6).round().astype(int).tolist()))
    evidence = {}
    for item in summaries[3:]:
        name = item["id"]
        visual = run / "visualizations" / name
        vm = read_json(visual / "metadata.json")
        if vm["video_sha256"] != checksum(visual / "comparison.mp4"):
            raise ValueError("Source video changed")
        for rec, expected in zip(vm["inputs"], [source["reference"] / "E3", source["reference"] / "E4", run / name]):
            if rec["metadata_sha256"] != checksum(expected / "metadata.json") or [rec["low"], rec["high"]] != source["limits"]:
                raise ValueError("Visual source or scale changed")
        centers = [item["largest_abs_T_difference_frame"], item["largest_prediction_difference_frame"]]
        starts = sorted(set(max(0, min(n - 6, center - 2)) for center in centers))
        windows = [list(range(start, min(start + 6, n))) for start in starts]
        wanted = set(overview + [t for win in windows for t in win])
        decoded, timestamps = {}, []
        with av.open(str(visual / "comparison.mp4")) as container:
            for i, frame in enumerate(container.decode(video=0)):
                timestamps.append(float(frame.pts * frame.time_base))
                if i in wanted:
                    decoded[i] = frame.to_image()
        if len(timestamps) != n or not np.allclose(timestamps, np.array(source["times"]) - source["times"][0], rtol=0, atol=2e-6):
            raise ValueError("Video count or timestamps changed")
        labels = ["RGB", "E3", "Full E4", name]
        name_dir = out / name
        name_dir.mkdir()
        make_sheet(decoded, overview, labels, width, height, source["times"], name_dir / "overview.png")
        for window in windows:
            make_sheet(decoded, window, labels, width, height, source["times"],
                       name_dir / f"consecutive_{window[0]}_{window[-1]}.png", roi)
        evidence[name] = {"source_video_sha256": vm["video_sha256"], "decoded_frames": n,
            "timestamps_verified": True, "overview_frames": overview, "consecutive_windows": windows,
            "roi_xyxy": roi, "selection": "Uniform overview; six-frame windows around the largest absolute per-pair T difference and largest mean raw prediction difference from full E4. Diagnostic selection, not random.",
            "images_sha256": {p.name: checksum(p) for p in sorted(name_dir.glob("*.png"))},
            "visual_inspection_status": "awaiting_assistant_inspection"}
    return evidence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--output-dir")
    p.add_argument("--roi", nargs=4, type=int, help="Original-grid x0 y0 x1 y1 for consecutive crops")
    a = p.parse_args()
    run = repo_path(a.run)
    out = repo_path(a.output_dir) if a.output_dir else run / "analysis"
    if out.exists():
        raise FileExistsError("Use a new analysis directory; previous reviews stay unchanged")
    meta, source, rows = load_verified(run)
    summaries, pairs = summarize(run, source, rows)
    out.mkdir(parents=True)
    figures = out / "figures"; figures.mkdir()
    # Uniform columns: additional selection fields are kept in JSON, not silently dropped.
    basic_keys = list(summaries[0])
    write_csv(out / "summary.csv", [{k: s[k] for k in basic_keys} for s in summaries])
    write_csv(out / "per_pair.csv", [row for table in pairs.values() for row in table])
    plot_summary(figures, summaries, pairs)
    visual = visual_evidence(run, figures, source, summaries, a.roi)
    result = {"status": "quantitative_complete_visual_pending", "clip_id": meta["clip_id"],
              "experiment": "confidence", "frames": len(source["times"]), "pairs": len(source["times"]) - 1,
              "normalization_limits": source["limits"], "parameters": meta["parameters"],
              "results": summaries, "visual_evidence": visual,
              "verification": "Current matching user-run fusion verification receipt checked; source hashes validated; seven raw T/L1/coverage/weight/departure diagnostics recomputed; four video timestamps checked.",
              "model_or_fusion_inference_performed": False,
              "source_metadata_sha256": checksum(run / "metadata.json"),
              "source_verification_sha256": checksum(run / "verification.json"),
              "analysis_script_sha256": checksum(Path(__file__)),
              "artifacts_sha256": {p.relative_to(out).as_posix(): checksum(p) for p in sorted(out.rglob("*")) if p.is_file()},
              "caveats": ["Fixed RGB/flow reference excludes some occlusion and boundary regions.",
                          "No depth GT; lower T is not accuracy or perceptual quality.",
                          "Departure from DA is not error; DA is not ground truth.",
                          "Frame pairs are correlated; no significance test.",
                          "Removing a soft term preserves hard gates and optical flow warping.",
                          "Historical and new runtime measurements are not a shared benchmark.",
                          "Post-hoc study; frozen main result is unchanged."]}
    write_json(out / "quantitative_review.json", result)
    print(json.dumps({"status": result["status"], "clip_id": meta["clip_id"], "results": summaries},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
