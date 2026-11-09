"""Generate LaTeX numbers/tables directly from measured JSON results."""
from pathlib import Path
import json

root=Path(__file__).resolve().parents[1]
report=root/'report';report.mkdir(exist_ok=True)
read=lambda name:json.loads((root/'results'/name).read_text())
nyu=read('nyu-baseline.json');stereo=read('stereo-baseline.json');final=read('final-evaluation.json')
runtime=read('runtime.json');efficient_runtime=read('runtime-efficient.json')
values={
    'NyuAbsRel':f"{nyu['metrics']['abs_rel']:.5f}",
    'NyuRMSE':f"{nyu['metrics']['rmse']:.5f}",
    'NyuDelta':f"{nyu['metrics']['delta1']:.5f}",
    'StereoMAE':f"{stereo['mean_pair_mae_px']:.3f}",
    'StereoCoverage':f"{100*stereo['mean_pair_coverage']:.2f}",
    'TemporalGain':f"{final['relative_temporal_proxy_reduction_percent']:.2f}",
    'EdgeRetention':f"{100*final['rows'][-1]['edge_retention']:.2f}",
    'OriginalFPS':f"{runtime['median_fps']:.2f}",
    'EfficientFPS':f"{efficient_runtime['median_fps']:.2f}",
    'SpeedGain':f"{100*(efficient_runtime['median_fps']/runtime['median_fps']-1):.1f}",
}
efficient_path=root/'results/efficient-evaluation.json'
if efficient_path.exists():
    efficient=json.loads(efficient_path.read_text())
    values['EfficientGain']=f"{efficient['relative_temporal_proxy_reduction_percent']:.2f}"
    values['EfficientEdge']=f"{100*efficient['rows'][-1]['edge_retention']:.2f}"
text='\n'.join('\\newcommand{\\'+name+'}{'+value+'}' for name,value in values.items())+'\n'
(report/'results.tex').write_text(text,encoding='utf-8')
lines=[r'\begin{tabular}{lrrr}',r'\toprule Method & Temporal & Edge (\%) & Step frames\\',r'\midrule']
for row in final['rows']:
    label={'baseline':'Frame-wise','ema':'EMA','flow':'Flow','occlusion':'Occlusion','adaptive':'Adaptive (selected)'}[row['method']]
    lines.append(f"{label} & {row['temporal']:.5f} & {100*row['edge_retention']:.1f} & {row['synthetic_settling_frames']}" + r" \\")
lines += [r'\bottomrule',r'\end{tabular}']
(report/'ablation-rows.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(text)
