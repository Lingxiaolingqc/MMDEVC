"""RGB Sobel/Canny x GT-depth Sobel/Canny; supplement, not model evaluation."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import html
import json
import time
import cv2
import numpy as np
import analyze_rgb_depth_edges as base
from p3_depth.evaluation.depth_edge_detectors import depth_gradients, depth_canny, DERIVATIVE_SCALE
from p3_depth.evaluation.rgb_depth_edges import (
    evaluation_domain, rgb_gradients, canny_mask, gt_edge_masks,
    top_count_mask, near_edges, score_association,
)

TARGETS = [.01, .02, .05]
PRIMARY = .02  # Fixed before this run; close to the earlier jump-boundary density.
TOLS = [1, 2, 3]
GRID = np.geomspace(1., 4096., 49)/DERIVATIVE_SCALE
GROUP = ['split', 'mode', 'detected_kind', 'detected_method', 'reference_method', 'depth_target', 'tolerance_px']
RENAME = {'rgb_edges':'detected_edges', 'gt_edges':'reference_edges',
          'matched_rgb':'matched_detected', 'matched_gt':'matched_reference',
          'random_matched_rgb':'random_matched_detected', 'random_matched_gt':'random_matched_reference',
          'rgb_only_near_pixels':'detected_only_near_pixels', 'depth_only_near_pixels':'reference_only_near_pixels'}


def calibrate(dataset, fits, output):
    hist = np.zeros(65536, np.int64)
    counts = np.zeros(len(GRID), np.int64)
    pixels = 0
    scenes = set()
    for i, index in enumerate(fits, 1):
        sample = dataset.load(index+1)
        scenes.add(sample.scene)
        _, domain, mag, derivatives = depth_gradients(sample.depth_m)
        bins = np.floor(mag[domain]*DERIVATIVE_SCALE).astype(int)
        if bins.max() >= len(hist): raise ValueError('Histogram overflow')
        hist += np.bincount(bins, minlength=len(hist))
        pixels += int(domain.sum())
        for k, high in enumerate(GRID):
            counts[k] += int(depth_canny(derivatives, float(high), domain).sum())
        if i % 100 == 0 or i == len(fits): print(f'DEPTH CALIBRATE {i}/{len(fits)}', flush=True)
    settings = []
    for target in TARGETS:
        threshold = float(np.searchsorted(np.cumsum(hist), pixels*(1-target))/DERIVATIVE_SCALE)
        settings.append(dict(method='sobel', target=target, high=threshold,
                             density_estimate=float(hist[int(threshold*DERIVATIVE_SCALE)+1:].sum()/pixels), reachable=True))
        k = int(np.argmin(abs(counts/pixels-target)))
        settings.append(dict(method='canny', target=target, high=float(GRID[k]),
                             density_estimate=float(counts[k]/pixels), reachable=bool(counts[-1]/pixels<=target<=counts[0]/pixels)))
    base.write_json(output/'depth_calibration.json', dict(settings=settings, pixels=pixels,
        canny_grid=GRID.tolist(), canny_densities=(counts/pixels).tolist(),
        selection='Fitting depth edge density only; no RGB-depth match scores used',
        primary_target=PRIMARY, primary_reason='Prespecified 2% depth density, near earlier 1.5% jump-edge density; not selected by validation F1',
        derivative_scale=DERIVATIVE_SCALE, sobel_quantile_bin_width=1/DERIVATIVE_SCALE))
    return settings, scenes


def masks(sample, rgb_settings, depth_settings):
    region, domain, mag, derivatives = depth_gradients(sample.depth_m)
    gray, rgb_mag = rgb_gradients(sample.rgb)
    rgb = {}
    for s in rgb_settings:
        rgb[s['detector']] = ((rgb_mag>s['threshold']) & domain if s['detector']=='sobel'
                              else canny_mask(gray, s['threshold'], domain))
    rgb['sobel_equal_count'] = top_count_mask(rgb_mag, domain, int(rgb['canny'].sum()))
    depth = {}
    for s in depth_settings:
        depth[(s['method'], s['target'])] = ((mag>s['high']) & domain if s['method']=='sobel'
                                             else depth_canny(derivatives, s['high'], domain))
    depth[('sobel_equal_count', PRIMARY)] = top_count_mask(mag, domain, int(depth[('canny',PRIMARY)].sum()))
    jump = gt_edge_masks(sample.depth_m, region, domain)[.1]
    return domain, rgb, depth, jump


def measure(sample, split, domain, left, right, mode, kind, lm, rm, target, tol):
    result = score_association(left, right, domain, tol)
    return dict(sample_id=sample.sample_id, matlab_index=sample.matlab_index, scene=sample.scene,
        split=split, mode=mode, detected_kind=kind, detected_method=lm, reference_method=rm,
        depth_target=target, tolerance_px=tol,
        **{RENAME.get(k,k):v for k,v in result.items()})


def analyze(dataset, indices, split, rgb_settings, depth_settings, fitting_scenes):
    rows, scenes = [], set()
    removed = 0
    for i, index in enumerate(indices, 1):
        sample = dataset.load(index+1)
        scenes.add(sample.scene)
        if split=='validation' and sample.scene in fitting_scenes: raise ValueError('Scene overlap')
        domain, rgb, depth, jump = masks(sample, rgb_settings, depth_settings)
        removed += int(evaluation_domain(sample.depth_m)[1].sum()-domain.sum())
        for tol in TOLS:
            for target in TARGETS:
                for lm in ('sobel','canny'):
                    for rm in ('sobel','canny'):
                        rows.append(measure(sample,split,domain,rgb[lm],depth[(rm,target)],'fixed','rgb',lm,rm,target,tol))
            for lm in ('sobel_equal_count','canny'):
                for rm in ('sobel_equal_count','canny'):
                    rows.append(measure(sample,split,domain,rgb[lm],depth[(rm,PRIMARY)],'equal_count','rgb',lm,rm,PRIMARY,tol))
            for method in ('sobel','canny','sobel_equal_count'):
                rows.append(measure(sample,split,domain,rgb[method],jump,'anchor','rgb',method,'jump10',0.,tol))
                rows.append(measure(sample,split,domain,depth[(method,PRIMARY)],jump,'anchor','depth',method,'jump10',PRIMARY,tol))
        if i % 100 == 0 or i == len(indices): print(f'CROSS SCORE {split} {i}/{len(indices)}', flush=True)
    return rows, scenes, removed


def aggregate(rows):
    keys=['domain_pixels','detected_edges','reference_edges','matched_detected','matched_reference',
          'random_matched_detected','random_matched_reference']
    sums={k:sum(r[k] for r in rows) for k in keys}
    p=sums['matched_detected']/sums['detected_edges'] if sums['detected_edges'] else 0.
    r=sums['matched_reference']/sums['reference_edges'] if sums['reference_edges'] else 0.
    rp=sums['random_matched_detected']/sums['detected_edges'] if sums['detected_edges'] else 0.
    rr=sums['random_matched_reference']/sums['reference_edges'] if sums['reference_edges'] else 0.
    return dict(images=len(rows),scenes=len({r['scene'] for r in rows}),**sums,
                precision=p,recall=r,f1=2*p*r/(p+r) if p+r else 0.,
                macro_f1=float(np.mean([r['f1'] for r in rows if r['f1'] is not None])),
                detected_density=sums['detected_edges']/sums['domain_pixels'],
                reference_density=sums['reference_edges']/sums['domain_pixels'],
                random_precision=rp,random_recall=rr,random_expected_count_f1=2*rp*rr/(rp+rr) if rp+rr else 0.)


def summarize(rows):
    groups=defaultdict(list)
    for r in rows: groups[tuple(r[k] for k in GROUP)].append(r)
    return [dict(zip(GROUP,key),**aggregate(rs)) for key,rs in sorted(groups.items())]


def bootstrap(rows):
    primary=[r for r in rows if r['split']=='validation' and r['mode']=='equal_count' and r['tolerance_px']==2]
    scenes=sorted({r['scene'] for r in primary})
    draws=np.random.default_rng(20261118).integers(0,len(scenes),(3000,len(scenes)))
    results=[]
    for rm in ('sobel_equal_count','canny'):
        arrays={}
        for lm in ('sobel_equal_count','canny'):
            subset=[r for r in primary if r['reference_method']==rm and r['detected_method']==lm]
            arrays[lm]=np.array([[sum(r[k] for r in subset if r['scene']==scene)
                          for k in ('detected_edges','reference_edges','matched_detected','matched_reference')] for scene in scenes],float)
        def scores(a):
            p=a[...,2]/a[...,0];r=a[...,3]/a[...,1]
            return np.stack((p,r,2*p*r/(p+r)),axis=-1)
        diff=scores(arrays['canny'][draws].sum(1))-scores(arrays['sobel_equal_count'][draws].sum(1))
        point=scores(arrays['canny'].sum(0))-scores(arrays['sobel_equal_count'].sum(0))
        results.append(dict(reference=rm,comparison='RGB Canny minus same-count RGB Sobel',
            metrics={k:dict(difference_pp=float(point[i]*100),interval_95_pp=(np.quantile(diff[:,i],[.025,.975])*100).tolist()) for i,k in enumerate(('precision','recall','f1'))}))
    return dict(scenes=len(scenes),replicates=3000,conditional_on='Fixed fitting calibration and depth definition',results=results)


def plots(summary, output):
    folder=output/'figures';folder.mkdir()
    for mode in ('fixed','equal_count'):
        sobel='sobel' if mode=='fixed' else 'sobel_equal_count'
        names=[sobel,'canny']
        selected=[r for r in summary if r['split']=='validation' and r['mode']==mode and r['depth_target']==PRIMARY and r['tolerance_px']==2]
        fig,axes=base.plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
        for ax,key in zip(axes,('precision','recall','f1')):
            mat=np.array([[next(r[key] for r in selected if r['detected_method']==lm and r['reference_method']==rm) for rm in names] for lm in names])
            im=ax.imshow(mat,vmin=0,vmax=1,cmap='viridis')
            for y in range(2):
                for x in range(2):ax.text(x,y,f'{mat[y,x]:.3f}',ha='center',va='center',color='white' if mat[y,x]<.6 else 'black',fontsize=15)
            ax.set(xticks=[0,1],xticklabels=['Depth Sobel','Depth Canny'],yticks=[0,1],yticklabels=['RGB Sobel','RGB Canny'],title=key)
        fig.colorbar(im,ax=axes);fig.suptitle(f'{mode}: validation 218 | RGB 5%, depth 2% fitting targets | 2px')
        fig.savefig(folder/f'matrix_{mode}.png',dpi=170);base.plt.close(fig)
    fig,axes=base.plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,key in zip(axes,('precision','recall','f1')):
        for lm in ('sobel','canny'):
            for rm in ('sobel','canny'):
                rs=sorted([r for r in summary if r['split']=='validation' and r['mode']=='fixed' and r['detected_method']==lm and r['reference_method']==rm and r['tolerance_px']==2],key=lambda r:r['depth_target'])
                ax.plot([r['depth_target']*100 for r in rs],[r[key] for r in rs],'-o',label=f'RGB {lm} / depth {rm}')
        ax.set(xlabel='Fitting depth edge density target (%)',ylabel=key);ax.grid(alpha=.2)
    axes[-1].legend(fontsize=8);fig.savefig(folder/'density_sensitivity.png',dpi=170);base.plt.close(fig)


def examples(dataset, selected, rgb_settings, depth_settings, rows, output):
    folder=output/'examples';folder.mkdir()
    records=[]
    for selection in selected:
        sid=selection['sample_id'];sample=dataset.load(int(sid.split('_')[-1]))
        domain,rgb,depth,jump=masks(sample,rgb_settings,depth_settings)
        fig,axes=base.plt.subplots(3,4,figsize=(18,10.8),constrained_layout=True)
        def show(ax,mask,title,color=(255,190,0)):
            img=sample.rgb.copy();img[mask]=color;ax.imshow(img);ax.set_title(title,fontsize=10)
        axes[0,0].imshow(sample.rgb);axes[0,0].set_title('RGB')
        axes[0,1].imshow(np.ma.masked_where(~domain,sample.depth_m),vmin=0,vmax=10,cmap='viridis');axes[0,1].set_title('GT depth, fixed 0-10 m')
        show(axes[0,2],rgb['sobel'],'RGB Sobel', (255,40,180));show(axes[0,3],rgb['canny'],'RGB Canny',(255,40,180))
        show(axes[1,0],jump,'Original depth jump >10%')
        show(axes[1,1],depth[('sobel',PRIMARY)],'GT log-depth Sobel')
        show(axes[1,2],depth[('canny',PRIMARY)],'GT log-depth Canny')
        axes[1,3].text(.02,.9,'Bottom: fixed-threshold 2 x 2\n\nGreen: matched RGB edge\nMagenta: unmatched RGB edge\nOrange: missed depth edge\n\nTolerance: 2 px\nDepth-edge definition varies\nNot a depth-accuracy ranking',va='top',fontsize=12)
        for ax,(lm,rm) in zip(axes[2],[(l,r) for l in ('sobel','canny') for r in ('sobel','canny')]):
            left,right=rgb[lm],depth[(rm,PRIMARY)]
            img=sample.rgb.copy();img[right&~near_edges(left,2)]=(255,190,0)
            img[left&~near_edges(right,2)]=(255,40,180);img[left&near_edges(right,2)]=(30,240,100)
            row=next(r for r in rows if r['sample_id']==sid and r['mode']=='fixed' and r['depth_target']==PRIMARY and r['tolerance_px']==2 and r['detected_method']==lm and r['reference_method']==rm)
            values=[f'{row[k]:.3f}' if row[k] is not None else 'NA' for k in ('precision','recall','f1')]
            ax.imshow(img);ax.set_title(f'RGB {lm} / depth {rm}\nP {values[0]}, R {values[1]}, F1 {values[2]}',fontsize=10)
        for ax in axes.flat:ax.axis('off')
        fig.suptitle(f'{sid} | {sample.scene} | Same cases selected in the previous diagnostic')
        name=f'examples/{sid}.png';fig.savefig(output/name,dpi=140);base.plt.close(fig)
        records.append(dict(sample_id=sid,scene=sample.scene,file=name))
    return records


def report(summary, selections, output):
    lines=['# RGB / GT 深度 Sobel × Canny 补充诊断','',
        '577 fitting 校准、218 validation 分析；GT 为原始 filled depth；test 未解码，未训练或运行深度模型。',
        '', '深度使用固定 log(m) 浮点梯度。Canny 接收 Sobel 梯度 ×512 的 int16 输入，未将深度量化为 8-bit 图片。共享 Gaussian 5×5 σ=1；Canny low/high=0.4。',
        '', 'RGB 使用上一轮 fitting 5% 目标的固定阈值；深度在 fitting 上以 1%/2%/5% 密度单独选阈值，主设置预定为 2%。没有按 RGB—深度重合成绩调阈值。实际密度见表，达不到目标的情况见 depth_calibration.json。',
        '', '在原评测域基础上排除深度无效值周围 4 px，避免平滑、梯度和非极大值抑制受无效值污染；所有方法及原跳变参照均在同一新域复算。',
        '', '## 四组交叉比较（2 px）','',
        '| 控制 | RGB 方法 | 深度方法 | RGB 密度 | 深度密度 | P | R | F1 |','|---|---|---|---:|---:|---:|---:|---:|']
    for r in summary:
        if r['split']=='validation' and r['mode']!='anchor' and r['depth_target']==PRIMARY and r['tolerance_px']==2:
            lines.append(f'| {r["mode"]} | {r["detected_method"]} | {r["reference_method"]} | {r["detected_density"]:.2%} | {r["reference_density"]:.2%} | {r["precision"]:.4f} | {r["recall"]:.4f} | {r["f1"]:.4f} |')
    lines+=['','equal_count：RGB Sobel 数量匹配 RGB Canny；深度 Sobel 数量匹配深度 Canny，分别在每张图精确控制。RGB 和深度两端数量不强行相等。',
        '', '## 共同参照：原深度跳变 >10%','', '| 输入 | 方法 | P | R | F1 |','|---|---|---:|---:|---:|']
    for r in summary:
        if r['split']=='validation' and r['mode']=='anchor' and r['tolerance_px']==2:
            lines.append(f'| {r["detected_kind"]} | {r["detected_method"]} | {r["precision"]:.4f} | {r["recall"]:.4f} | {r["f1"]:.4f} |')
    lines+=['', '## 图表与边界','', '![固定阈值](figures/matrix_fixed.png)','', '![数量控制](figures/matrix_equal_count.png)','', '![密度敏感性](figures/density_sensitivity.png)',
        '', 'P 为检出的 RGB 边缘在深度边缘附近的比例，R 为深度边缘被 RGB 边缘覆盖的比例；anchor 表相应以其输入作为检出端。F1 是两者调和平均，均不是深度预测准确率。',
        '', '改变深度算子会改变参照，不能只比较对角线后宣称算法更好。深度梯度也可能来自连续斜面；细化和平滑会改变边缘宽度及位置。阈值和数量控制不能完全去除这些表示差异。',
        '', '本补充为查看上一轮结果后设计的开发分析。保留原结果，不将本轮的较高一致性包装成模型改进。', '', '## 相同案例']
    for e in selections:lines+=['',f'### {e["sample_id"]} / {e["scene"]}','',f'![{e["sample_id"]}]({e["file"]})']
    (output/'analysis.md').write_text('\n'.join(lines),encoding='utf-8')
    pictures=['figures/matrix_fixed.png','figures/matrix_equal_count.png','figures/density_sensitivity.png']+[e['file'] for e in selections]
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>RGB-depth 2x2</title><style>body{max-width:1600px;margin:24px auto;font:18px sans-serif}img{width:100%;margin:16px 0}</style><h1>RGB / GT 深度 Sobel × Canny</h1><p>同样的 577 / 218 开发划分，未用 test。<a href="analysis.md">指标、协议与限制</a>。</p>'+''.join(f'<img src="{html.escape(x)}">' for x in pictures),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'outputs/part3_improvement/edge_association/nyuv2_sobel_canny_20261118')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/part3_improvement/edge_association/nyuv2_sobel_canny_cross_20261118')
    p.add_argument('--data-root',type=Path,default=ROOT/'data/nyuv2')
    p.add_argument('--split',type=Path,default=ROOT/'configs/edge_association_split.json')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use a fresh output path')
    config=json.loads(a.split.read_text(encoding='utf-8'));fits,vals=base.validate_split(config,a.data_root)
    source=json.loads((a.source/'metadata.json').read_text(encoding='utf-8'))
    if source['status']!='complete' or source['split']!=config:raise ValueError('Source split/status mismatch')
    for name in ('calibration.json','example_selection.json'):
        if base.sha(a.source/name)!=source['artifacts'][name]:raise ValueError('Changed source artifact')
    rgb_settings=[s for s in json.loads((a.source/'calibration.json').read_text())['settings'] if s['target_density']==.05]
    selected=json.loads((a.source/'example_selection.json').read_text())
    if any(int(s['sample_id'].split('_')[-1])-1 not in vals for s in selected):raise ValueError('Example outside validation')
    a.output.mkdir(parents=True);start=time.perf_counter();cv2.setNumThreads(1)
    files=[Path(__file__).resolve(),Path(base.__file__).resolve(),ROOT/'src/p3_depth/evaluation/depth_edge_detectors.py',ROOT/'src/p3_depth/evaluation/rgb_depth_edges.py',ROOT/'src/p3_depth/temporal/fusion.py']
    meta=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),split=config,
        source_directory=str(a.source.resolve()),source_metadata_sha256=base.sha(a.source/'metadata.json'),rgb_settings=rgb_settings,
        protocol=dict(depth_transform='log meters, no per-image scaling',smoothing='Gaussian 5x5 sigma 1',derivative_scale=DERIVATIVE_SCALE,primary_depth_target=PRIMARY,depth_targets=TARGETS,tolerances=TOLS),
        implementation_sha256={str(f.relative_to(ROOT)):base.sha(f) for f in files},versions=dict(opencv=cv2.__version__,numpy=np.__version__))
    (a.output/'source_at_run').mkdir()
    for f in files:(a.output/'source_at_run'/(f.name+'.txt')).write_bytes(f.read_bytes())
    base.write_json(a.output/'metadata.json',meta)
    with base.NYUv2Labeled(a.data_root/'nyu_depth_v2_labeled.mat') as ds:
        settings,fit_scenes=calibrate(ds,fits,a.output)
        rows,fs,removed_fit=analyze(ds,fits,'fitting',rgb_settings,settings,fit_scenes)
        vr,vs,removed_val=analyze(ds,vals,'validation',rgb_settings,settings,fit_scenes);rows+=vr
        if fs&vs:raise ValueError('Scene overlap')
        summary=summarize(rows)
        base.write_csv(a.output/'per_image.csv',rows);base.write_json(a.output/'summary.json',summary);base.write_csv(a.output/'summary.csv',summary)
        base.write_json(a.output/'paired_scene_bootstrap.json',bootstrap(rows));plots(summary,a.output)
        example_records=examples(ds,selected,rgb_settings,settings,vr,a.output)
        base.write_json(a.output/'example_selection.json',example_records)
    report(summary,example_records,a.output)
    meta.update(status='complete',elapsed_seconds=time.perf_counter()-start,images_read=dict(fitting=577,validation=218,test=0),scene_counts=dict(fitting=len(fs),validation=len(vs)),
        excluded_additional_pixels=dict(fitting=removed_fit,validation=removed_val),rows=len(rows),
        artifacts={f.relative_to(a.output).as_posix():base.sha(f) for f in sorted(a.output.rglob('*')) if f.is_file() and f.name!='metadata.json'})
    base.write_json(a.output/'metadata.json',meta)
    print(json.dumps(dict(status='PASS',images=795,test_images_read=0,rows=len(rows),seconds=meta['elapsed_seconds'],output=str(a.output)),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
