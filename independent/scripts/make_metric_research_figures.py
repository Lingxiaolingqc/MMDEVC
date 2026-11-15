"""Render frozen-test metric-depth comparisons with shared GT error scales."""
import csv
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from depth_project.data import load_nyu_sample
from depth_project.metric_training import write_json


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/metric-research-final"
OUT = ROOT / "results/metric-research/figures"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "savefig.dpi": 240})


def save(figure, name):
    figure.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def read_rows(name):
    with (RUN / f"{name}-per-image.csv").open(encoding="utf-8") as stream:
        return [{"image_id": int(row["image_id"]), "scene": row["scene"], "abs_rel": float(row["abs_rel"])}
                for row in csv.DictReader(stream)]


def sample(index):
    rgb, truth = load_nyu_sample(index, ROOT / "data/nyu/nyu_depth_v2_labeled.mat")
    with np.load(RUN / f"prediction_{index:04d}.npz", allow_pickle=False) as packet:
        baseline, final = packet["baseline_m"], packet["selected_m"]
    valid = np.zeros(truth.shape, bool)
    valid[45:471, 41:601] = True
    valid &= np.isfinite(truth) & (truth > 0.001) & (truth < 10)
    errors = []
    for depth in [baseline, final]:
        error = np.full(truth.shape, np.nan)
        error[valid] = np.abs(np.clip(depth[valid], 0.001, 10) - truth[valid]) / truth[valid]
        errors.append(error)
    return rgb, truth, baseline, final, errors


def pipeline():
    selection = json.loads((ROOT / "results/metric-research/seed-selection.json").read_text())
    figure, axis = plt.subplots(figsize=(7.2, 2.15))
    axis.set(xlim=(0, 10), ylim=(0, 3))
    axis.axis("off")
    boxes = [(0.1, 1.1, 1.2, "RGB image\n480 x 640"),
             (1.8, 1.1, 2.4, "UniDepthV2-S\n147,456-parameter LoRA"),
             (4.7, 1.1, 2.2, "Frozen scale\nFlip + seed averaging"),
             (7.5, 1.1, 2.3, "Metric prediction\nDepth in meters")]
    for left, bottom, width, label in boxes:
        axis.add_patch(FancyBboxPatch((left, bottom), width, 0.8, boxstyle="round,pad=.05",
                                     facecolor="#edf2f5", edgecolor="#536b7a", linewidth=0.8))
        axis.text(left + width / 2, bottom + 0.4, label, ha="center", va="center")
    for left, right in [(1.3, 1.8), (4.2, 4.7), (6.9, 7.5)]:
        axis.add_patch(FancyArrowPatch((left, 1.5), (right, 1.5), arrowstyle="-|>",
                                      mutation_scale=12, color="#536b7a"))
    axis.text(5, 2.55, "Whole-scene CV → three seeds → fixed all-795 refit → frozen 654-image test",
              ha="center", fontsize=9)
    axis.text(5, 0.55, f"Rank-8 Q/V adapters   |   25% restored-flip weight   |   {len(selection['chosen_seeds'])} seeds",
              ha="center", fontsize=8)
    axis.text(5, 0.1, "Development uses RGB and measured depth. Test prediction consumes RGB.", ha="center")
    save(figure, "research-pipeline.png")


def comparisons(baseline, final):
    gains = np.array([before["abs_rel"] - after["abs_rel"] for before, after in zip(baseline, final)])
    order = np.argsort(gains, kind="stable")
    positions = [int(order[-1]), int(order[len(order) // 2]), int(order[0])]
    rules = ["Largest gain", "Median change", "Largest regression"]
    selected = []
    figure, axes = plt.subplots(3, 6, figsize=(10.5, 5.4))
    for row, (position, rule) in enumerate(zip(positions, rules)):
        index = final[position]["image_id"]
        rgb, truth, original, improved, errors = sample(index)
        axes[row, 0].imshow(rgb)
        for column, depth in enumerate([truth, original, improved], 1):
            depth_plot = axes[row, column].imshow(depth, cmap="inferno", vmin=0, vmax=10)
        for column, error in enumerate(errors, 4):
            error_plot = axes[row, column].imshow(np.ma.masked_invalid(error), cmap="magma", vmin=0, vmax=0.5)
        axes[row, 0].set_title(f"{rule}\nNYU {index}", fontsize=11)
        axes[row, 4].set_title(f"AbsRel {baseline[position]['abs_rel']:.3f}", fontsize=11)
        axes[row, 5].set_title(f"AbsRel {final[position]['abs_rel']:.3f}", fontsize=11)
        selected.append({"rule": rule, "image_id": index, "scene": final[position]["scene"],
                         "baseline_abs_rel": baseline[position]["abs_rel"], "selected_abs_rel": final[position]["abs_rel"]})
    for axis, label in zip(axes[0], ["RGB", "Ground truth", "UniDepthV2-S", "Final", "Baseline error", "Final error"]):
        axis.text(0.5, 1.37, label, transform=axis.transAxes, ha="center", fontsize=12)
    for axis in axes.flat:
        axis.axis("off")
    figure.subplots_adjust(left=0.01, right=0.99, top=0.85, bottom=0.1, hspace=0.36, wspace=0.025)
    figure.colorbar(depth_plot, cax=figure.add_axes([0.22, 0.035, 0.38, 0.015]), orientation="horizontal",
                    label="Depth in meters, shared 0-10 range")
    figure.colorbar(error_plot, cax=figure.add_axes([0.69, 0.035, 0.28, 0.015]), orientation="horizontal",
                    label="Pixel AbsRel, capped at 0.5")
    for axis in figure.axes[-2:]:
        axis.tick_params(labelsize=11)
        axis.xaxis.label.set_size(11)
    save(figure, "research-comparison.png")
    return selected


def hard_cases(final):
    hardest = sorted(final, key=lambda row: (-row["abs_rel"], row["image_id"]))[:3]
    figure, axes = plt.subplots(3, 4, figsize=(7.2, 5.3))
    for row, record in enumerate(hardest):
        rgb, truth, _, prediction, errors = sample(record["image_id"])
        axes[row, 0].imshow(rgb)
        for column, depth in enumerate([truth, prediction], 1):
            axes[row, column].imshow(depth, cmap="inferno", vmin=0, vmax=10)
        axes[row, 3].imshow(np.ma.masked_invalid(errors[1]), cmap="magma", vmin=0, vmax=0.5)
        axes[row, 0].set_title(f"NYU {record['image_id']}", fontsize=8)
        axes[row, 3].set_title(f"AbsRel {record['abs_rel']:.3f}", fontsize=8)
    for axis, label in zip(axes[0], ["RGB", "Ground truth (m)", "Final depth (m)", "Pixel AbsRel"]):
        axis.text(0.5, 1.22, label, transform=axis.transAxes, ha="center", fontsize=9)
    for axis in axes.flat:
        axis.axis("off")
    figure.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.02, hspace=0.25, wspace=0.025)
    save(figure, "research-hard-cases.png")
    return hardest


def application(index):
    rgb, _, baseline, final, _ = sample(index)
    blurred = cv2.GaussianBlur(rgb, (0, 0), 8)
    images = [rgb]
    for depth in [baseline, final]:
        alpha = np.clip((3.2 - depth) / 0.4, 0, 1)[:, :, None]
        images.append(np.rint(alpha * rgb + (1 - alpha) * blurred).astype(np.uint8))
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.05))
    for axis, image, label in zip(axes, images, ["RGB", "Baseline depth blur", "Final depth blur"]):
        axis.imshow(image)
        axis.set_title(label)
        axis.axis("off")
    figure.tight_layout(pad=0.2)
    save(figure, "research-application.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    baseline, final = read_rows("baseline"), read_rows("selected")
    if len(final) != 654 or [row["image_id"] for row in baseline] != [row["image_id"] for row in final]:
        raise ValueError("Research figures require the paired complete test evaluation")
    pipeline()
    selected = comparisons(baseline, final)
    hardest = hard_cases(final)
    application(selected[0]["image_id"])
    write_json(ROOT / "results/metric-research/figure-selection.json", {"comparisons": selected, "hard_cases": hardest})


if __name__ == "__main__":
    main()
