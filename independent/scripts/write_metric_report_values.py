"""Write report numbers and ablation rows from the completed test evaluation."""
import json
from pathlib import Path
import shutil


root = Path(__file__).resolve().parents[1]
summary = json.loads((root / 'results/metric-improvement.json').read_text())
baseline = summary['metrics']['baseline']
final = summary['metrics']['residual_refiner']
values = {
    'MetricBaseAbs': f"{baseline['abs_rel']:.5f}",
    'MetricFinalAbs': f"{final['abs_rel']:.5f}",
    'MetricBaseRMSE': f"{baseline['rmse']:.5f}",
    'MetricFinalRMSE': f"{final['rmse']:.5f}",
    'MetricBaseDelta': f"{baseline['delta1']:.5f}",
    'MetricFinalDelta': f"{final['delta1']:.5f}",
    'MetricGain': f"{summary['abs_rel_reduction_percent']:.2f}",
    'MetricCILow': f"{summary['scene_cluster_bootstrap_95_percent_reduction_interval'][0]:.2f}",
    'MetricCIHigh': f"{summary['scene_cluster_bootstrap_95_percent_reduction_interval'][1]:.2f}",
    'MetricReproduction': f"{summary['maximum_reproduced_baseline_pixel_difference_m']:.2g}" if summary['maximum_reproduced_baseline_pixel_difference_m'] is not None else 'unavailable',
    'MetricImproved': str(summary['per_image_improved_count']),
    'MetricWorsened': str(summary['per_image_worsened_count']),
    'MetricBaseMS': f"{1000*summary['timing']['baseline_median_s']:.1f}",
    'MetricFinalMS': f"{1000*summary['timing']['improved_median_s']:.1f}",
    'MetricMemory': f"{summary['timing']['peak_allocated_mib']:.0f}",
}
values['MetricReproductionSentence'] = (
    'Reproduced and original baseline predictions differ by at most '+values['MetricReproduction']+' m per pixel.'
    if summary['maximum_reproduced_baseline_pixel_difference_m'] is not None
    else 'Historical baseline arrays were unavailable for an additional pixelwise comparison.'
)
(root / 'report/metric-results.tex').write_text('\n'.join('\\newcommand{\\'+key+'}{'+value+'}' for key, value in values.items())+'\n', encoding='utf-8')
lines = [r'\begin{tabular}{lrrr}', r'\toprule Method & AbsRel & RMSE (m) & $\delta_1$\\', r'\midrule']
labels = {'baseline': 'Unchanged UniDepthV2-S', 'global_scale': 'Global scale only', 'flip_only': 'Flip averaging only', 'selected_simple': 'Selected flip + scale', 'residual_refiner': 'Residual refinement'}
for key, label in labels.items():
    row = summary['metrics'][key]
    lines.append(f"{label} & {row['abs_rel']:.5f} & {row['rmse']:.5f} & {row['delta1']:.5f}"+r'\\')
lines += [r'\bottomrule', r'\end{tabular}']
(root / 'report/metric-ablation-rows.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')
for name in labels:
    shutil.copyfile(root / f'runs/metric-final/{name}-per-image.csv', root / f'results/metric-{name}-per-image.csv')
print(json.dumps(values, indent=2))
