"""Generate measured NYU comparison figures with ground truth and shared scales."""
import csv
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

from depth_project.data import load_nyu_sample


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/figures'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'savefig.dpi': 240, 'axes.spines.top': False, 'axes.spines.right': False})


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def records(name):
    with (ROOT / f'runs/metric-final/{name}-per-image.csv').open() as stream:
        return [{key: (int(value) if key in ['image_id', 'valid_count'] else value if key == 'scene' else float(value)) for key, value in row.items()} for row in csv.DictReader(stream)]


def pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 2.1))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis('off')
    boxes = [(.1, 1.1, 1.3, .8, 'RGB image\n480 x 640'), (2., 1.1, 2., .8, 'Frozen UniDepthV2-S\nOriginal + flipped RGB'), (4.6, 1.1, 1.5, .8, 'Flip blend\nGlobal scale'), (6.7, 1.1, 1.5, .8, 'Small residual\nrefinement CNN'), (8.8, 1.1, 1., .8, 'Metric depth\nin meters')]
    for x, y, w, h, label in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=.06', fc='#eef0f5', ec='#687383', lw=.8))
        ax.text(x+w/2, y+h/2, label, ha='center', va='center', fontsize=8)
    for start, end in [(1.4, 1.95), (4, 4.55), (6.1, 6.65), (8.2, 8.75)]:
        ax.add_patch(FancyArrowPatch((start, 1.5), (end, 1.5), arrowstyle='-|>', mutation_scale=12, color='#576473'))
    ax.text(5, .4, 'Development labels fit calibration and the refiner; test inference uses RGB and model outputs.', ha='center', fontsize=8)
    ax.text(5, 2.5, 'Same Small checkpoint and foundation-model input resolution throughout', ha='center', weight='bold', fontsize=10)
    save(fig, 'metric-pipeline.png')


def comparisons():
    baseline, improved = records('baseline'), records('residual_refiner')
    gains = np.array([a['abs_rel']-b['abs_rel'] for a, b in zip(baseline, improved)])
    ordered = np.argsort(gains)
    indices = [int(ordered[-1]), int(ordered[len(ordered)//2]), int(ordered[0])]
    selections = []
    fig, axes = plt.subplots(3, 6, figsize=(10, 5.5))
    titles = ['RGB', 'Ground truth (m)', 'UniDepthV2-S (m)', 'Improved (m)', 'Baseline AbsRel', 'Improved AbsRel']
    for row, position in enumerate(indices):
        index = baseline[position]['image_id']
        rgb, gt = load_nyu_sample(index, ROOT / 'data/nyu/nyu_depth_v2_labeled.mat')
        with np.load(ROOT / f'runs/metric-final/prediction_{index:04d}.npz') as source:
            original, final = source['baseline_m'], source['depth_m']
        valid = np.zeros(gt.shape, bool)
        valid[45:471, 41:601] = True
        valid &= np.isfinite(gt) & (gt > .001) & (gt < 10)
        errors = []
        for prediction in [original, final]:
            error = np.full(gt.shape, np.nan)
            error[valid] = np.abs(np.clip(prediction[valid], .001, 10)-gt[valid])/gt[valid]
            errors.append(error)
        axes[row, 0].imshow(rgb)
        for col, values in enumerate([gt, original, final], 1):
            depth_plot = axes[row, col].imshow(values, cmap='inferno', vmin=0, vmax=10)
        for col, error in enumerate(errors, 4):
            error_plot = axes[row, col].imshow(np.ma.masked_invalid(error), cmap='magma', vmin=0, vmax=.5)
        for ax in axes[row]: ax.axis('off')
        label = ['Largest gain', 'Median change', 'Largest regression'][row]
        axes[row, 0].set_title(f'{label}\nNYU {index}', fontsize=8)
        axes[row, 4].set_title(f"{baseline[position]['abs_rel']:.3f}", fontsize=8)
        axes[row, 5].set_title(f"{improved[position]['abs_rel']:.3f}", fontsize=8)
        selections.append({'rule': label, 'image_id': index, 'baseline_abs_rel': baseline[position]['abs_rel'], 'improved_abs_rel': improved[position]['abs_rel']})
    for ax, title in zip(axes[0], titles): ax.text(.5, 1.35, title, transform=ax.transAxes, ha='center', fontsize=8)
    fig.subplots_adjust(left=.02, right=.98, bottom=.09, top=.85, hspace=.38, wspace=.025)
    depth_axis = fig.add_axes([.21, .035, .4, .015])
    error_axis = fig.add_axes([.68, .035, .29, .015])
    fig.colorbar(depth_plot, cax=depth_axis, orientation='horizontal', label='Depth (m), shared 0-10 scale')
    fig.colorbar(error_plot, cax=error_axis, orientation='horizontal', label='Pixel AbsRel, clipped at 0.5')
    save(fig, 'metric-comparison.png')
    (ROOT / 'results/metric-figure-selection.json').write_text(json.dumps(selections, indent=2), encoding='utf-8')

    index = selections[0]['image_id']
    rgb, _ = load_nyu_sample(index, ROOT / 'data/nyu/nyu_depth_v2_labeled.mat')
    with np.load(ROOT / f'runs/metric-final/prediction_{index:04d}.npz') as source:
        depths = [source['baseline_m'], source['depth_m']]
    blurred = cv2.GaussianBlur(rgb, (0, 0), 8)
    panels = []
    for depth in depths:
        alpha = np.clip((2.2-depth)/.4, 0, 1)[:, :, None]
        panels.append(np.rint(alpha*rgb+(1-alpha)*blurred).astype(np.uint8))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.1))
    for ax, image, title in zip(axes, [rgb, *panels], ['RGB', 'Baseline depth blur', 'Improved depth blur']):
        ax.imshow(image); ax.set_title(title); ax.axis('off')
    fig.tight_layout(pad=.2)
    save(fig, 'metric-application.png')


def quantitative():
    summary = json.loads((ROOT / 'results/metric-improvement.json').read_text())
    names = ['baseline', 'flip_only', 'global_scale', 'selected_simple', 'residual_refiner']
    labels = ['Unchanged', 'Flip only', 'Scale only', 'Flip + scale', 'Residual refiner']
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, metric, title in zip(axes, ['abs_rel', 'rmse', 'delta1'], ['AbsRel (lower)', 'RMSE, m (lower)', 'Delta1 (higher)']):
        values = [summary['metrics'][name][metric] for name in names]
        ax.bar(range(5), values, color=['#939aa4']*4+['#586fa6'])
        ax.set_xticks(range(5), labels, rotation=40, ha='right', fontsize=7)
        ax.set_title(title); ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
    fig.tight_layout(pad=.5, w_pad=1.2)
    save(fig, 'metric-ablation.png')
    base, final = records('baseline'), records('residual_refiner')
    fig, ax = plt.subplots(figsize=(3.5, 3))
    ax.scatter([row['abs_rel'] for row in base], [row['abs_rel'] for row in final], s=8, alpha=.45, color='#586fa6')
    ax.plot([0, .6], [0, .6], color='#777', lw=.8)
    ax.set(xlabel='Baseline per-image AbsRel', ylabel='Improved per-image AbsRel', xlim=(0, .6), ylim=(0, .6))
    ax.set_title('Points below the diagonal improved')
    fig.tight_layout(); save(fig, 'metric-paired-errors.png')


if __name__ == '__main__':
    pipeline(); comparisons(); quantitative()
