"""One bounded, matched E4 relaxation study; preserve all earlier results."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import gc
from time import perf_counter
import cv2
import numpy as np
from _common import ROOT, checksum, write_json, write_csv
from _video_review import read_json, contact_sheet, verify_preview
from _media import video_preview, checked_comparison
from part3_improvement.diagnose_confidence_regions import source_for
from part3_improvement.run_supplementary_ablations import describe
from p3_depth.temporal.refinement import refinement_sequence, fixed_regions
from p3_depth.temporal.fusion import CONFIDENCE_TERMS

SETTING = ROOT / "configs/fusion_refinement_20261109.json"
MODES = ("original", "global", "selective")
IMPLEMENTATION = ["scripts/part3_improvement/run_confidence_refinement.py",
                  "src/p3_depth/temporal/refinement.py", "src/p3_depth/temporal/fusion.py"]


def spec(mode):
    return dict(id=mode, variant="E4", history_source="recursive", confidence_terms=list(CONFIDENCE_TERMS))


def folder_for(s):
    return ROOT / "outputs/part3_improvement/confidence_refinement" / s["cm"]["split"] / s["cm"]["clip_id"]


def verify(s, out, setting):
    m = read_json(out / "metadata.json")
    if checksum(SETTING) != m["setting_sha256"] or checksum(s["reference"] / "metadata.json") != m["source_suite_sha256"]:
        raise ValueError("Configuration or original suite changed")
    for name, digest in m["implementation_sha256"].items():
        if checksum(ROOT / name) != digest:
            raise ValueError("Implementation changed: " + name)
    for name, digest in m["artifacts_sha256"].items():
        if checksum(out / name) != digest:
            raise ValueError("Artifact changed: " + name)
    rows = []
    for mode in MODES:
        run = out / mode
        saved = np.load(run / "depths.npy", mmap_mode="r", allow_pickle=False)
        w = np.load(run / "history_weight.npy", mmap_mode="r", allow_pickle=False)
        pred, weights = refinement_sequence(s["base"], s["valid"], s["sm"]["config"]["parameters"], s["fields"], setting, mode)
        if not np.array_equal(pred, saved, equal_nan=True) or not np.array_equal(w, weights):
            raise ValueError("Saved recursion/weights do not recompute: " + mode)
        valid = np.load(run / "valid.npy", allow_pickle=False)
        if not np.array_equal(valid, np.isfinite(pred)):
            raise ValueError("Validity differs")
        _, scores, _ = describe(pred, valid, weights, spec(mode), s, 0, False)
        if scores != read_json(run / "metrics.json"):
            raise ValueError("Saved scores do not recompute")
        preview = verify_preview(s["cache"], run, s["times"], s["limits"])
        rows.append(dict(mode=mode, raw_and_weights_exact=True, scores_exact=True, preview=preview))
        del pred, weights, valid
    old = np.load(s["reference"] / "E4/depths.npy", mmap_mode="r", allow_pickle=False)
    if not np.array_equal(old, np.load(out / "original/depths.npy", mmap_mode="r", allow_pickle=False), equal_nan=True):
        raise ValueError("Original E4 changed")
    write_json(out / "verification.json", dict(status="verified", metadata_sha256=checksum(out / "metadata.json"), modes=rows,
               original_E4_exact=True, scope="Recomputed saved arrays, weights, metrics and decoded presentation timestamps"))
    print("PASS refinement raw-array/weight/metric/timestamp recomputation:", out, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clip", choices=("dev_tokyo_100", "course_rollercoaster_70"), required=True)
    p.add_argument("--verify-only", action="store_true")
    a = p.parse_args()
    s = source_for(a.clip); setting = read_json(SETTING); out = folder_for(s)
    cv2.setNumThreads(8)
    if a.verify_only:
        verify(s, out, setting); return
    if (out / "metadata.json").exists() or any((out / mode).exists() for mode in MODES):
        raise FileExistsError("Preserve existing refinement study")
    decision = None
    if s["cm"]["split"] == "test":
        decision_path = out.parents[1] / "dev/dev_tokyo_100/decision.json"
        decision = read_json(decision_path)
        if decision["status"] != "dev_decision_frozen" or decision["setting_sha256"] != checksum(SETTING):
            raise ValueError("Freeze a matching dev decision before the post-hoc course comparison")
        if checksum(decision_path.parent / "analysis/review.json") != decision["dev_review_sha256"]:
            raise ValueError("Frozen dev review changed")
    out.mkdir(parents=True, exist_ok=True)
    par = s["sm"]["config"]["parameters"]
    # Warm every path on the same two frames. Rotate the timed order between repeats.
    small_flow = {k: v[:1] for k, v in s["fields"].items()}
    for mode in MODES:
        refinement_sequence(s["base"][:2], s["valid"][:2], par, small_flow, setting, mode)
    timings = []
    for rep in range(3):
        order = MODES[rep:] + MODES[:rep]
        for position, mode in enumerate(order):
            gc.collect()
            start = perf_counter()
            pred, weights = refinement_sequence(s["base"], s["valid"], par, s["fields"], setting, mode)
            elapsed = perf_counter() - start
            run = out / mode
            if rep == 0:
                run.mkdir()
                for name, arr in (("depths", pred), ("valid", np.isfinite(pred)), ("history_weight", weights)):
                    np.save(run / (name + ".npy"), arr)
            else:
                for name, arr in (("depths", pred), ("history_weight", weights)):
                    if not np.array_equal(arr, np.load(run / (name + ".npy"), mmap_mode="r", allow_pickle=False), equal_nan=True):
                        raise ValueError("Nondeterministic repeat")
            timings.append(dict(repeat=rep, order=position, mode=mode, fusion_seconds=elapsed))
            print(f"PASS {mode} repeat {rep+1}: {elapsed:.3f}s", flush=True)
            del pred, weights
    write_csv(out / "timings.csv", timings)
    rows = []
    for mode in MODES:
        run = out / mode
        pred = np.load(run / "depths.npy", mmap_mode="r", allow_pickle=False)
        weights = np.load(run / "history_weight.npy", mmap_mode="r", allow_pickle=False)
        valid = np.load(run / "valid.npy", mmap_mode="r", allow_pickle=False)
        seconds = [r["fusion_seconds"] for r in timings if r["mode"] == mode]
        median = float(np.median(seconds))
        row, scores, pairs = describe(pred, valid, weights, spec(mode), s, median, False)
        # Do not mix fresh CPU measurements with historical model/flow times.
        for key in ("estimated_pipeline_fps", "flow_seconds"):
            row.pop(key)
        row.update(fusion_seconds_min=min(seconds), fusion_seconds_max=max(seconds), repeats=3)
        rows.append(row)
        write_json(run / "metrics.json", scores); write_csv(run / "per_pair.csv", pairs)
        write_json(run / "metadata.json", dict(status="complete", model={"model": "E4_" + mode},
            cache_metadata=s["cm"], prediction_type="inverse_depth_like", unit="arbitrary", alignment="none", gt_used=False,
            normalization_reference={"limits": s["limits"], "source_metadata_sha256": checksum(s["baseline"] / "metadata.json")},
            **{name+"_sha256": checksum(run / (name+".npy")) for name in ("depths", "valid", "history_weight")},
            parameters=par, setting=setting, mode=mode, fusion_seconds=median, config_status="bounded_posthoc",
            source_suite_sha256=checksum(s["reference"] / "metadata.json")))
        video_preview(s["cache"], run)
    write_csv(out / "comparison.csv", rows)
    if len({r["reference_pixels"] for r in rows}) != 1:
        raise ValueError("Mismatched denominator")
    checked_comparison(s["cache"], [s["baseline"], *[out / mode for mode in MODES]], out / "visualizations")
    rgb = np.load(s["cache"] / "frames.npy", mmap_mode="r", allow_pickle=False)
    preds = [np.load(out / mode / "depths.npy", mmap_mode="r", allow_pickle=False) for mode in MODES]
    masks = [np.load(out / mode / "valid.npy", mmap_mode="r", allow_pickle=False) for mode in MODES]
    figures = out / "figures"; figures.mkdir()
    contact_sheet(rgb, preds, masks, [s["limits"]]*3, list(MODES), s["times"],
                  np.linspace(0, len(rgb)-1, 6).round().astype(int).tolist(), figures / "overview.png")
    roi = [200, 60, 390, 350] if s["cm"]["split"] == "dev" else [100, 60, 450, 260]
    starts = [28, 73] if s["cm"]["split"] == "dev" else [14, 32, 64]
    span = s["limits"][1] - s["limits"][0]
    per_frame = []
    for t in range(len(rgb)):
        changes = [float(np.abs(d[t]-preds[0][t]).mean()/span) for d in preds[1:]]
        per_frame.append(dict(frame=t, global_change_from_original=changes[0], selective_change_from_original=changes[1]))
    most = max(per_frame, key=lambda r: r["selective_change_from_original"])["frame"]
    starts.append(min(max(0, most-2), len(rgb)-6))
    for start in sorted(set(starts)):
        contact_sheet(rgb, preds, masks, [s["limits"]]*3, list(MODES), s["times"], list(range(start, start+6)),
                      figures / f"roi_{start}_{start+5}.png", roi)
    write_csv(out / "per_frame_change.csv", per_frame)
    region_rows = []
    weights = [np.load(out / mode / "history_weight.npy", mmap_mode="r", allow_pickle=False) for mode in MODES]
    acc = {(mode, group): np.zeros(4, dtype=np.float64) for mode in MODES for group in ("reliable", "edge_band", "other")}
    for t in range(1, len(rgb)):
        f = {k: s["fields"][k][t-1] for k in ("backward", "forward", "photo_residual", "fb_residual")}
        reg = fixed_regions(s["base"][t], f, par, setting)
        for group, mask in dict(reliable=reg["reliable"], edge_band=reg["valid"] & reg["edge_band"],
                               other=reg["valid"] & ~reg["edge_band"] & ~reg["reliable"]).items():
            for i, mode in enumerate(MODES):
                acc[mode, group] += [mask.sum(), weights[i][t][mask].astype(np.float64).sum(),
                    (np.abs(preds[i][t][mask].astype(np.float64)-s["base"][t][mask])/span).sum(),
                    (np.abs(preds[i][t][mask].astype(np.float64)-preds[0][t][mask])/span).sum()]
    for (mode, group), (n, w, departure, change) in acc.items():
        region_rows.append(dict(mode=mode, region=group, pixels=int(n), mean_history_weight=w/n if n else None,
                               mean_departure_from_DA=departure/n if n else None, mean_change_from_original=change/n if n else None))
    write_csv(out / "regions.csv", region_rows)
    write_json(out / "metadata.json", dict(status="complete", split=s["cm"]["split"], clip=a.clip,
        setting=setting, setting_sha256=checksum(SETTING), dev_decision=decision,
        source_suite_sha256=checksum(s["reference"] / "metadata.json"),
        source_verification_sha256=checksum(s["reference"] / "verification.json"),
        timing_protocol="CPU only, OpenCV threads=8; two-frame warmup per mode; three repeats with rotated mode order; excludes I/O/render/scoring. Includes allocation and per-mode implementation overhead, not optimized theoretical complexity.",
        interpretation="One predeclared factor and guard; current-DA-derived fixed masks; no GT; all modes recursive. Outside guard retains the original rule for the candidate's own history, not necessarily identical output to original. No test retuning.",
        visual_sampling=dict(uniform_full_frames=6, roi=roi, starts=sorted(set(starts)), max_change_frame=most,
                             max_change_selection="largest whole-frame selective-versus-original normalized absolute difference"),
        implementation_sha256={name: checksum(ROOT / name) for name in IMPLEMENTATION},
        artifacts_sha256={p.relative_to(out).as_posix(): checksum(p) for p in out.rglob("*") if p.is_file() and "diagnosis" not in p.relative_to(out).parts}))
    verify(s, out, setting)


if __name__ == "__main__":
    main()
