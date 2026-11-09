"""Audit saved E4 confidence by fixed current-frame regions before refinement."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from _common import ROOT, repo_path, checksum, write_json, write_csv
from _video_review import read_json
from part3_improvement.run_supplementary_ablations import load_sources
from p3_depth.temporal.refinement import fixed_regions
from p3_depth.temporal.flow import sample_previous
from p3_depth.temporal.fusion import fuse_step, CONFIDENCE_TERMS


def source_for(clip):
    split = "dev" if clip == "dev_tokyo_100" else "test"
    cache = ROOT / "outputs/shared/video_cache" / clip
    base = ROOT / "outputs/part1_baseline/dav2_relative" / split / clip
    flow = ROOT / "outputs/part3_improvement/flow" / split / clip
    reference = (ROOT / "outputs/part3_improvement/dev_candidates/alpha070" / clip if split == "dev"
                 else ROOT / "outputs/part3_improvement/test" / clip)
    return load_sources(cache, base, flow, reference)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clip", choices=("dev_tokyo_100", "course_rollercoaster_70"), default="dev_tokyo_100")
    p.add_argument("--config", default="configs/fusion_refinement_20261109.json")
    a = p.parse_args()
    setting = read_json(repo_path(a.config))
    s = source_for(a.clip)
    out = ROOT / "outputs/part3_improvement/confidence_refinement" / s["cm"]["split"] / a.clip / "diagnosis"
    if out.exists():
        raise FileExistsError("Preserve diagnosis")
    par = s["sm"]["config"]["parameters"]
    old = np.load(s["reference"] / "E4/depths.npy", mmap_mode="r", allow_pickle=False)
    weights = np.load(s["reference"] / "E4/history_weight.npy", mmap_mode="r", allow_pickle=False)
    names = ("all_accepted", "reliable_interior", "reliable_positive", "edge_band",
             "other_accepted", "hard_rejected")
    sums = {g: dict(pixels=0, finite_component_pixels=0, weight_sum=0., components=np.zeros(4),
                    dominant=np.zeros(4), depth_exponent_over_one=0) for g in names}
    per_frame = []
    chosen = (30, 75) if s["cm"]["split"] == "dev" else (15, 35)
    maps = []
    for t in range(1, len(old)):
        cur = s["base"][t]
        f = {k: s["fields"][k][t-1] for k in ("backward", "forward", "photo_residual", "fb_residual")}
        regions = fixed_regions(cur, f, par, setting)
        warped, _ = sample_previous(old[t-1], f["backward"], np.isfinite(old[t-1]))
        pred, w = fuse_step(cur, old[t-1], "E4", par, representation="inverse_depth_like", **f)
        if not np.array_equal(w, weights[t]) or not np.array_equal(pred, old[t], equal_nan=True):
            raise ValueError("Original E4 no longer reproduces")
        terms = np.stack([f["photo_residual"] / par["photo_sigma"],
                          f["fb_residual"] / par["flow_sigma_px"],
                          np.abs(warped-cur) / max(regions["span"], 1e-8) / par["depth_span_sigma"],
                          regions["edge"] / par["edge_span_sigma"]])
        finite = np.isfinite(terms).all(axis=0)
        winner = np.argmax(np.where(np.isfinite(terms), terms, -np.inf), axis=0)
        nonzero = np.sum(np.where(np.isfinite(terms), terms, 0), axis=0) > 1e-8
        hard, reliable, band = regions["hard"], regions["reliable"], regions["edge_band"]
        groups = dict(all_accepted=hard, reliable_interior=reliable,
                      reliable_positive=reliable & (cur > 0), edge_band=hard & band,
                      other_accepted=hard & ~band & ~reliable,
                      hard_rejected=regions["valid"] & ~hard)
        for name, mask in groups.items():
            count = int(mask.sum()); support = mask & finite; fc = int(support.sum())
            acc = sums[name]; acc["pixels"] += count; acc["finite_component_pixels"] += fc
            acc["weight_sum"] += float(w[mask].astype(np.float64).sum())
            acc["components"] += terms[:, support].astype(np.float64).sum(axis=1)
            acc["dominant"] += [int((support & nonzero & (winner == k)).sum()) for k in range(4)]
            acc["depth_exponent_over_one"] += int((support & (terms[2] > 1)).sum())
            per_frame.append({"frame": t, "group": name, "pixels": count,
                              "mean_weight": float(w[mask].mean()) if count else None,
                              "mean_depth_exponent": float(terms[2, support].mean()) if fc else None})
        if t in chosen:
            maps.append((t, reliable.copy(), terms[2].copy(), w.copy()))
    rows = []
    total = (len(old)-1)*old.shape[1]*old.shape[2]
    for name, acc in sums.items():
        n, f = acc["pixels"], acc["finite_component_pixels"]
        rows.append({"group": name, "pixels": n, "fraction_all_pairs": n/total,
                     "mean_history_weight": acc["weight_sum"]/n if n else None,
                     "finite_component_pixels": f,
                     **{"mean_"+term+"_exponent": acc["components"][i]/f if f else None
                        for i, term in enumerate(CONFIDENCE_TERMS)},
                     **{"dominant_"+term+"_fraction": acc["dominant"][i]/f if f else None
                        for i, term in enumerate(CONFIDENCE_TERMS)},
                     "depth_exponent_over_one_fraction": acc["depth_exponent_over_one"]/f if f else None})
    out.mkdir(parents=True); (out / "figures").mkdir()
    write_csv(out / "regions.csv", rows); write_csv(out / "per_frame.csv", per_frame)
    show = [r for r in rows if r["group"] in ("reliable_interior", "reliable_positive", "edge_band", "other_accepted")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
    x = np.arange(len(show))
    for i, term in enumerate(CONFIDENCE_TERMS):
        axes[0].bar(x + (i-1.5)*.2, [r["mean_"+term+"_exponent"] for r in show], width=.2, label=term)
    axes[0].set_ylabel("Mean exponent contribution")
    axes[0].legend()
    axes[1].bar(x, [r["mean_history_weight"] for r in show])
    axes[1].set(ylim=(0, .7), ylabel="Mean saved E4 history weight")
    for ax in axes:
        ax.set_xticks(x, [r["group"].replace("_", "\n") for r in show], fontsize=8)
    fig.savefig(out / "figures/region_terms.png", dpi=180); plt.close(fig)
    rgb = np.load(s["cache"] / "frames.npy", mmap_mode="r", allow_pickle=False)
    fig, axes = plt.subplots(len(maps), 4, figsize=(12, 3*len(maps)), layout="constrained")
    for axes_row, (t, reliable, depth_term, weight) in zip(axes, maps):
        overlay = np.array(rgb[t], copy=True)
        overlay[reliable] = (.5*overlay[reliable] + .5*np.array([0, 255, 255])).astype(np.uint8)
        images = (rgb[t], overlay, np.clip(depth_term, 0, 2), weight)
        for j, (ax, image, title) in enumerate(zip(axes_row, images, ("RGB", "Cyan = reliable interior", "Depth exponent [0,2]", "Saved weight [0,.7]"))):
            ax.imshow(image, **({"vmin": 0, "vmax": 2 if j==2 else .7, "cmap": "viridis"} if j>1 else {}))
            ax.set_title(f"f{t}: {title}", fontsize=10); ax.axis("off")
    fig.savefig(out / "figures/region_maps.png", dpi=170); plt.close(fig)
    write_json(out / "metadata.json", {
        "status": "diagnosis_complete", "clip": a.clip, "setting": setting,
        "setting_sha256": checksum(repo_path(a.config)),
        "source_suite_sha256": checksum(s["reference"] / "metadata.json"),
        "source_verification_sha256": checksum(s["reference"] / "verification.json"),
        "all_original_E4_frames_and_weights_recomputed": True,
        "interpretation": "Regions overlap as declared: reliable_positive is a subset, not semantic foreground. Exponents are suppression strengths, not error probabilities. All-zero components have no dominant term. Hard-rejected residual means use only finite components.",
        "selected_frames": list(chosen), "baseline_modified": False,
        "implementation_sha256": {n: checksum(ROOT / n) for n in
            ("scripts/part3_improvement/diagnose_confidence_regions.py", "src/p3_depth/temporal/refinement.py")},
        "artifacts_sha256": {p.relative_to(out).as_posix(): checksum(p) for p in out.rglob("*") if p.is_file()}
    })
    print(rows, flush=True)
    print("PASS saved confidence region diagnosis:", out, flush=True)


if __name__ == "__main__":
    main()

