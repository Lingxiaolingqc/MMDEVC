"""Add verified compact evidence to the separate streaming/refinement sample report."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import shutil
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from _common import ROOT, checksum, write_json
from _video_review import read_json, read_rows, contact_sheet
from p3_depth.application import background_blur

OUT = ROOT / "report/sample_20261109_extension"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-draft", action="store_true")
    args = parser.parse_args()
    if (OUT / "extension_assets.json").exists() and not args.refresh_draft:
        raise FileExistsError("Preserve completed extension asset build")
    data = OUT / "extension"; data.mkdir(exist_ok=True)
    sources, copies = {}, {}

    def track(path):
        sources[path.relative_to(ROOT).as_posix()] = checksum(path)
        return path

    def copy(path, name):
        shutil.copyfile(track(path), data / name)
        copies[name] = checksum(data / name)

    for split, clip in (("dev", "dev_tokyo_100"), ("test", "course_rollercoaster_70")):
        modes = ROOT / "outputs/part2_models/vda_modes" / split / clip
        study = ROOT / "outputs/part3_improvement/confidence_refinement" / split / clip
        for label, analysis in (("streaming", modes / "analysis"), ("refinement", study / "analysis")):
            review = read_json(track(analysis / "review.json"))
            if not review["status"].startswith("complete_for_declared") or not review["independent_reanalysis"]:
                raise ValueError("Incomplete independent review")
            if checksum(analysis / "analysis.md") != review["analysis_markdown_sha256"]:
                raise ValueError("Changed narrative")
            for name in ("review.json", "analysis.md"):
                copy(analysis / name, f"{label}_{split}_{name}")
        for name in ("comparison.csv", "quantitative_review.json", "per_pair.csv"):
            copy(modes / "analysis" / name, f"streaming_{split}_{name}")
        for mode in ("offline", "streaming"):
            for name in ("metadata.json", "timing_repeats.csv"):
                copy(modes / mode / name, f"streaming_{split}_{mode}_{name}")
            if mode == "streaming":
                copy(modes / mode / "per_frame_latency.csv", f"streaming_{split}_per_frame_latency.csv")
        for name in ("comparison.csv", "regions.csv", "timings.csv", "metadata.json", "verification.json"):
            copy(study / name, f"refinement_{split}_{name}")
        receipt = read_json(study / "verification.json")
        if receipt["status"] != "verified" or receipt["metadata_sha256"] != checksum(study / "metadata.json"):
            raise ValueError("Refinement verification changed")
        for name in ("regions.csv", "metadata.json"):
            copy(study / "diagnosis" / name, f"diagnosis_{split}_{name}")
        if split == "dev":
            copy(study / "decision.json", "dev_decision.json")
        for source_meta in (read_json(study / "metadata.json"), read_json(modes / "analysis/quantitative_review.json")):
            base = study if "setting" in source_meta else modes / "analysis"
            for name, sha in source_meta["artifacts_sha256"].items():
                if checksum(track(base / name)) != sha:
                    raise ValueError("Changed reviewed artifact: " + name)
    copy(ROOT / "configs/fusion_refinement_20261109.json", "setting.json")
    clip = "course_rollercoaster_70"
    cache = ROOT / "outputs/shared/video_cache" / clip
    frames = np.load(track(cache / "frames.npy"), mmap_mode="r", allow_pickle=False)
    times = [float(r["timestamp_s"]) for r in read_rows(track(cache / "frames.csv"))]
    parent = ROOT / "outputs/part2_models/vda_modes/test" / clip
    predictions = [np.load(track(parent / mode / "depths.npy"), mmap_mode="r", allow_pickle=False) for mode in ("offline", "streaming")]
    masks = [np.load(track(parent / mode / "valid.npy"), mmap_mode="r", allow_pickle=False) for mode in ("offline", "streaming")]
    spans = [list(map(float, np.quantile(p[v], [.01, .99]))) for p, v in zip(predictions, masks)]
    contact_sheet(frames, predictions, masks, spans, ["offline", "streaming"], times, [0, 2],
                  OUT / "images/streaming.png", roi=[100, 60, 450, 260])
    parent = ROOT / "outputs/part3_improvement/confidence_refinement/test" / clip
    predictions = [np.load(track(parent / mode / "depths.npy"), mmap_mode="r", allow_pickle=False) for mode in ("original", "global", "selective")]
    masks = [np.load(track(parent / mode / "valid.npy"), mmap_mode="r", allow_pickle=False) for mode in ("original", "global", "selective")]
    span = read_json(track(parent / "original/metadata.json"))["normalization_reference"]["limits"]
    contact_sheet(frames, predictions, masks, [span]*3, ["original", "global", "selective"], times, [15, 35],
                  OUT / "images/refinement.png", roi=[100, 60, 450, 260])
    base = ROOT / "outputs/part1_baseline/dav2_relative/test" / clip
    da = np.load(track(base / "depths.npy"), mmap_mode="r", allow_pickle=False)
    valid = np.load(track(base / "valid.npy"), mmap_mode="r", allow_pickle=False)
    t = 35
    panels = [frames[t]]
    for pred, mask, transition in ((da, valid, (.45, .70)), (da, valid, (.10, .30)),
                                   (predictions[0], masks[0], (.10, .30))):
        panels.append(background_blur(frames[t], pred[t], mask[t], span, sigma=5, transition=transition)[0])
    fig, axes = plt.subplots(2, 2, figsize=(5, 3.2), layout="constrained")
    for ax, panel, title in zip(axes.flat, panels, ("RGB | frame 35", "DA | old thresholds", "DA | new thresholds", "E4 | new thresholds")):
        ax.imshow(panel[50:270, 70:500]); ax.set_title(title, fontsize=10); ax.axis("off")
    fig.savefig(OUT / "images/application_compact.png", dpi=200); plt.close(fig)
    track(ROOT / "report/sample_20261109/report_provenance.json")
    write_json(OUT / "extension_assets.json", dict(status="verified_sources", source_sha256=sources, copies_sha256=copies,
        images_sha256={n: checksum(OUT / "images" / n) for n in ("streaming.png", "refinement.png", "application_compact.png")},
        selection="Streaming initialization f0/f2; refinement prior inspected motion f15/f35; each source review records wider samples.",
        implementation_sha256=checksum(Path(__file__))))
    print("PASS separate report extension evidence and fixed-scale figures")


if __name__ == "__main__":
    main()
