"""Diagnose Sobel/Canny association with NYUv2 GT boundaries, development only."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import platform
import subprocess
import time

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from p3_depth.data.nyuv2 import NYUv2Labeled, read_standard_split
from p3_depth.evaluation.rgb_depth_edges import (
    evaluation_domain, rgb_gradients, canny_mask, gt_edge_masks,
    disk, near_edges, top_count_mask, score_association,
)

TARGETS = [.01, .02, .05, .10]
JUMPS = [.05, .10, .20]
TOLERANCES = [1, 2, 3]
CANNY_GRID = np.geomspace(2., 1024., 37)
COLORS = {'sobel': '#3366a0', 'canny': '#cf7423', 'sobel_equal_count': '#328667'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_split(config, data):
    fits, vals, tests = [config[k] for k in ('fit_ids', 'validation_ids', 'test_ids')]
    if config['index_base'] != 0 or [len(x) for x in (fits, vals, tests)] != [577, 218, 654]:
        raise ValueError('Need the pinned 577/218/654 zero-based split')
    if any(len(set(x)) != len(x) for x in (fits, vals, tests)) or set(fits)&set(vals):
        raise ValueError('Duplicate/overlapping development IDs')
    standard = read_standard_split(data/'splits.mat')
    if set(fits+vals) != set((standard['train']-1).tolist()) or set(tests) != set((standard['test']-1).tolist()):
        raise ValueError('Official split mismatch')
    for filename, field in [('nyu_depth_v2_labeled.mat', 'dataset_sha256'), ('splits.mat', 'split_sha256')]:
        if sha(data/filename) != config[field]:
            raise ValueError('Source hash mismatch: '+filename)
    return fits, vals


def calibrate(dataset, fits, output):
    hist = np.zeros(8192, np.int64)
    canny_counts = np.zeros(len(CANNY_GRID), np.int64)
    count = 0
    scenes = {}
    for number, index in enumerate(fits, 1):
        sample = dataset.load(index+1)
        scenes[index] = sample.scene
        _, domain = evaluation_domain(sample.depth_m)
        gray, magnitude = rgb_gradients(sample.rgb)
        bins = np.floor(magnitude[domain]*4).astype(np.int32)
        if bins.max() >= len(hist):
            raise ValueError('Gradient histogram overflow')
        hist += np.bincount(bins, minlength=len(hist))
        count += int(domain.sum())
        for i, threshold in enumerate(CANNY_GRID):
            canny_counts[i] += int(canny_mask(gray, float(threshold), domain).sum())
        if number % 100 == 0 or number == len(fits):
            print(f'CALIBRATE {number}/{len(fits)} fitting images', flush=True)
    cumulative = np.cumsum(hist)
    settings = []
    for target in TARGETS:
        threshold = float(np.searchsorted(cumulative, count*(1-target))/4)
        settings.append(dict(detector='sobel', target_density=target, threshold=threshold,
                             calibration_density_approx=float(hist[int(threshold*4)+1:].sum()/count),
                             target_reachable=True))
        i = int(np.argmin(np.abs(canny_counts/count-target)))
        settings.append(dict(detector='canny', target_density=target, threshold=float(CANNY_GRID[i]),
                             calibration_density_approx=float(canny_counts[i]/count),
                             target_reachable=bool(canny_counts[-1]/count <= target <= canny_counts[0]/count)))
    result = dict(selection='Only fitting RGB density; no GT-edge matching scores used',
                  settings=settings, pixels=count, canny_grid=CANNY_GRID.tolist(),
                  canny_densities=(canny_counts/count).tolist(),
                  sobel_histogram_quantization='quarter gradient unit; actual density reported during scoring',
                  smoothing='shared 5x5 Gaussian sigma=1 on uint8 gray',
                  canny='apertureSize=3, L2gradient=True, low=0.4*high',
                  sobel='float32 gradient magnitude, aperture=3, threshold > high')
    write_json(output/'calibration.json', result)
    return settings, scenes


def masks_for(rgb, domain, settings):
    gray, magnitude = rgb_gradients(rgb)
    masks = []
    for setting in settings:
        edges = (magnitude > setting['threshold']) & domain if setting['detector']=='sobel' else canny_mask(gray, setting['threshold'], domain)
        masks.append((setting, edges))
    canny = next(mask for setting, mask in masks if setting['detector']=='canny' and setting['target_density']==.05)
    masks.append((dict(detector='sobel_equal_count', target_density=.05, threshold=None),
                  top_count_mask(magnitude, domain, int(canny.sum()))))
    return masks


def analyze(dataset, indices, split, settings, fitting_scenes):
    rows, scenes = [], set()
    for number, index in enumerate(indices, 1):
        sample = dataset.load(index+1)
        scenes.add(sample.scene)
        if split=='validation' and sample.scene in fitting_scenes:
            raise ValueError('Scene leakage')
        region, domain = evaluation_domain(sample.depth_m)
        truth = gt_edge_masks(sample.depth_m, region, domain)
        neighborhoods = {tol: cv2.filter2D(domain.astype(np.float32), -1, disk(tol).astype(np.float32), borderType=cv2.BORDER_CONSTANT) for tol in TOLERANCES}
        gt_near = {(jump, tol): near_edges(truth[jump], tol) for jump in JUMPS for tol in TOLERANCES}
        for setting, mask in masks_for(sample.rgb, domain, settings):
            combinations = [(jump, tol) for jump in JUMPS for tol in TOLERANCES] if setting['target_density']==.05 else [(.1, 2)]
            for jump, tolerance in combinations:
                result = score_association(mask, truth[jump], domain, tolerance, gt_near[(jump,tolerance)],
                                           np.rint(neighborhoods[tolerance][truth[jump]]).astype(int))
                rows.append(dict(sample_id=sample.sample_id, matlab_index=index+1, source_index=index,
                                 scene=sample.scene, split=split, **setting, relative_jump=jump,
                                 tolerance_px=tolerance, **result))
        if number % 100==0 or number==len(indices):
            print(f'SCORE {split} {number}/{len(indices)} images', flush=True)
    return rows, sorted(scenes)


SUM_FIELDS = ['domain_pixels','rgb_edges','gt_edges','matched_rgb','matched_gt',
              'random_matched_rgb','random_matched_gt','both_near_pixels',
              'rgb_only_near_pixels','depth_only_near_pixels','neither_near_pixels']


def aggregate(rows):
    sums = {name: sum(row[name] for row in rows) for name in SUM_FIELDS}
    p = sums['matched_rgb']/sums['rgb_edges'] if sums['rgb_edges'] else 0.
    r = sums['matched_gt']/sums['gt_edges'] if sums['gt_edges'] else 0.
    rp = sums['random_matched_rgb']/sums['rgb_edges'] if sums['rgb_edges'] else 0.
    rr = sums['random_matched_gt']/sums['gt_edges'] if sums['gt_edges'] else 0.
    defined = [x['f1'] for x in rows if x['f1'] is not None]
    return dict(images=len(rows), scenes=len({x['scene'] for x in rows}), **sums,
                precision=p, recall=r, f1=2*p*r/(p+r) if p+r else 0.,
                macro_f1=float(np.mean(defined)) if defined else None,
                density=sums['rgb_edges']/sums['domain_pixels'],
                gt_density=sums['gt_edges']/sums['domain_pixels'],
                random_precision=rp, random_recall=rr,
                random_expected_count_f1=2*rp*rr/(rp+rr) if rp+rr else 0.,
                precision_lift_over_uniform_random=p/rp if rp else None)


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['split'],row['detector'],row['target_density'],row['relative_jump'],row['tolerance_px'])].append(row)
    summaries = []
    for (split, detector, target, jump, tolerance), group in sorted(groups.items()):
        summaries.append(dict(split=split, detector=detector, target_density=target,
                              relative_jump=jump, tolerance_px=tolerance, **aggregate(group)))
    return summaries


def paired_bootstrap(rows):
    selected = [r for r in rows if r['split']=='validation' and r['target_density']==.05 and r['relative_jump']==.1 and r['tolerance_px']==2]
    by_method = {m: {r['sample_id']: r for r in selected if r['detector']==m} for m in ('canny','sobel_equal_count')}
    if set(by_method['canny']) != set(by_method['sobel_equal_count']):
        raise ValueError('Paired images differ')
    for sid in by_method['canny']:
        if by_method['canny'][sid]['rgb_edges'] != by_method['sobel_equal_count'][sid]['rgb_edges']:
            raise ValueError('Equal-count control does not match')
    scenes = sorted({r['scene'] for r in by_method['canny'].values()})
    keys = ['rgb_edges','gt_edges','matched_rgb','matched_gt']
    totals = {}
    for m, values in by_method.items():
        totals[m] = np.array([[sum(r[k] for r in values.values() if r['scene']==scene) for k in keys] for scene in scenes], dtype=float)
    rng = np.random.default_rng(20261118)
    draws = rng.integers(0, len(scenes), size=(3000,len(scenes)))
    def metrics(x):
        p = np.divide(x[...,2],x[...,0],out=np.zeros(x.shape[:-1]),where=x[...,0]>0)
        r = np.divide(x[...,3],x[...,1],out=np.zeros(x.shape[:-1]),where=x[...,1]>0)
        f = np.divide(2*p*r,p+r,out=np.zeros_like(p),where=p+r>0)
        return np.stack([p,r,f],axis=-1)
    differences = metrics(totals['canny'][draws].sum(axis=1))-metrics(totals['sobel_equal_count'][draws].sum(axis=1))
    point = metrics(totals['canny'].sum(axis=0))-metrics(totals['sobel_equal_count'].sum(axis=0))
    return dict(scope='Paired scene bootstrap on validation, Canny minus exact-count Sobel; fixed protocol',
                scenes=len(scenes), images=len(by_method['canny']), replicates=3000,
                results={key:dict(difference_pp=float(point[i]*100),interval_95_pp=(np.quantile(differences[:,i],[.025,.975])*100).tolist()) for i,key in enumerate(['precision','recall','f1'])})


def make_plots(summaries, output):
    visual = output/'figures'
    visual.mkdir()
    primary = [r for r in summaries if r['split']=='validation' and r['relative_jump']==.1 and r['tolerance_px']==2]
    fig, axes = plt.subplots(1, 2, figsize=(12,4.5), constrained_layout=True)
    for method in ('sobel','canny'):
        data = sorted([r for r in primary if r['detector']==method],key=lambda r:r['target_density'])
        axes[0].plot([r['recall'] for r in data],[r['precision'] for r in data],'-o',label=method,color=COLORS[method])
        axes[1].plot([r['density']*100 for r in data],[r['f1'] for r in data],'-o',label=method,color=COLORS[method])
        for r in data:
            axes[0].annotate(f"{r['target_density']:.0%}",(r['recall'],r['precision']),xytext=(4,4),textcoords='offset points',fontsize=8)
    axes[0].set(xlabel='GT boundary recall',ylabel='RGB edge precision',title='Fixed thresholds selected on fitting RGB only')
    axes[1].set(xlabel='Actual validation RGB edge density (%)',ylabel='Boundary F1',title='Report density as well as matching quality')
    for ax in axes: ax.grid(alpha=.2);ax.legend()
    fig.suptitle('NYUv2 validation: 218 images | 10% GT depth jump | 2 px tolerance')
    fig.savefig(visual/'association_curves.png',dpi=180);plt.close(fig)
    labels=['Sobel fixed','Canny fixed','Sobel exact count']
    selected=[next(r for r in primary if r['detector']==m and r['target_density']==.05) for m in ('sobel','canny','sobel_equal_count')]
    fig,ax=plt.subplots(figsize=(10,4.5),constrained_layout=True)
    for i,key in enumerate(['precision','recall','f1']):
        bars=ax.bar(np.arange(3)+(i-1)*.25,[r[key] for r in selected],width=.25,label=key)
        ax.bar_label(bars,fmt='%.3f',fontsize=9)
    ax.set(xticks=np.arange(3),xticklabels=labels,ylim=(0,1),ylabel='Score',title='5% fitting-density setting; exact-count control matches Canny per image')
    ax.legend();fig.savefig(visual/'primary_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,method in zip(axes,('sobel','canny','sobel_equal_count')):
        matrix=np.array([[next(r['f1'] for r in summaries if r['split']=='validation' and r['detector']==method and r['target_density']==.05 and r['relative_jump']==jump and r['tolerance_px']==tol) for tol in TOLERANCES] for jump in JUMPS])
        im=ax.imshow(matrix,vmin=0,vmax=1,cmap='viridis')
        for y in range(3):
            for x in range(3): ax.text(x,y,f'{matrix[y,x]:.3f}',ha='center',va='center',color='white' if matrix[y,x]<.6 else 'black')
        ax.set(xticks=range(3),xticklabels=TOLERANCES,yticks=range(3),yticklabels=['5%','10%','20%'],xlabel='Match tolerance (px)',ylabel='GT relative jump',title=method)
    fig.colorbar(im,ax=axes,label='F1');fig.savefig(visual/'sensitivity.png',dpi=180);plt.close(fig)


def make_examples(dataset, rows, settings, output):
    primary=[r for r in rows if r['split']=='validation' and r['relative_jump']==.1 and r['tolerance_px']==2 and r['target_density']==.05]
    canny={r['sample_id']:r for r in primary if r['detector']=='canny'}
    sobel={r['sample_id']:r for r in primary if r['detector']=='sobel_equal_count'}
    ranked=sorted(canny.values(),key=lambda r:(r['f1'] if r['f1'] is not None else -1,r['sample_id']))
    usable=[r for r in ranked if r['gt_edges']>=100 and r['rgb_edges']>=100]
    choices=[('high_agreement',ranked[-1]),('median_agreement',ranked[len(ranked)//2]),
             ('low_precision',min(usable,key=lambda r:r['precision'])),('low_recall',min(usable,key=lambda r:r['recall'])),
             ('canny_advantage',max(usable,key=lambda r:r['f1']-sobel[r['sample_id']]['f1'])),
             ('sobel_advantage',min(usable,key=lambda r:r['f1']-sobel[r['sample_id']]['f1']))]
    folder=output/'examples';folder.mkdir()
    selected=[]
    for label,row in choices:
        sample=dataset.load(row['matlab_index']);rgb=sample.rgb
        region,domain=evaluation_domain(sample.depth_m)
        gt=gt_edge_masks(sample.depth_m,region,domain)[.1]
        masks={s['detector']:m for s,m in masks_for(rgb,domain,settings) if s['target_density']==.05}
        near_gt=near_edges(gt,2)
        def overlay(mask,color):
            image=rgb.copy();image[mask]=color;return image
        def marked(mask):
            image=rgb.copy();image[mask&near_gt]=(30,240,100);image[mask&~near_gt]=(255,40,180);return image
        fig,axes=plt.subplots(2,3,figsize=(15,8.4),constrained_layout=True)
        axes[0,0].imshow(rgb);axes[0,0].set_title('RGB')
        dep=axes[0,1].imshow(np.ma.masked_where(~region,sample.depth_m),vmin=0,vmax=10,cmap='viridis')
        axes[0,1].set_title('Filled GT depth (m)');fig.colorbar(dep,ax=axes[0,1],shrink=.75)
        axes[0,2].imshow(overlay(gt,(255,190,0)));axes[0,2].set_title('GT depth boundaries: orange')
        axes[1,0].imshow(marked(masks['sobel_equal_count']));axes[1,0].set_title('Sobel, same edge count as Canny')
        axes[1,1].imshow(marked(masks['canny']));axes[1,1].set_title('Canny: green matched / magenta unmatched')
        missed=gt & ~near_edges(masks['canny'],2)
        axes[1,2].imshow(overlay(missed,(255,190,0)));axes[1,2].set_title('GT boundaries missed by Canny: orange')
        for ax in axes.flat:ax.axis('off')
        fig.suptitle(f"{label} | {sample.sample_id} | {sample.scene}\nCanny P={row['precision']:.3f}, R={row['recall']:.3f}, F1={row['f1']:.3f}; Sobel matched-count F1={sobel[row['sample_id']]['f1']:.3f}")
        path=folder/f'{label}_{sample.sample_id}.png';fig.savefig(path,dpi=160);plt.close(fig)
        selected.append(dict(category=label,sample_id=sample.sample_id,scene=sample.scene,file=path.relative_to(output).as_posix(),
                             canny_precision=row['precision'],canny_recall=row['recall'],canny_f1=row['f1'],sobel_equal_count_f1=sobel[row['sample_id']]['f1']))
    write_json(output/'example_selection.json',selected)
    return selected


def report(summaries, bootstrap, examples, output):
    primary=[r for r in summaries if r['split']=='validation' and r['relative_jump']==.1 and r['tolerance_px']==2 and r['target_density']==.05]
    lines=['# RGB—深度边缘关联诊断：NYUv2 开发集','',
           '使用固定 577 张 fitting / 218 张场景不重叠 validation。没有读取 654 张 test 的 RGB 或 GT，没有训练或运行深度模型。',
           '', '## 主结果','',
           'GT 为 filled depths；四邻域最大 log-depth 跳变 > log(1.1)，保留两端边界像素。课程裁剪内严格有效 GT，经 3×3 腐蚀确定固定域；欧氏 2 px 匹配。',
           '', 'RGB 两法共享灰度化、5×5 Gaussian σ=1、3×3 Sobel。Canny 固定 low/high=0.4，L2 gradient；Sobel 使用连续梯度幅值阈值。',
           '', '阈值按 fitting RGB 的目标检出密度选择，不使用 GT 边界匹配成绩。下表为目标 5% 的固定阈值；实际 validation 密度可能不同。',
           '', '| 方法 | 实际边缘密度 | Precision | Recall | F1 | 随机 Precision | Precision/随机 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for r in primary:
        lines.append(f"| {r['detector']} | {r['density']:.2%} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['random_precision']:.4f} | {r['precision_lift_over_uniform_random']:.2f}× |")
    lines += ['', '`sobel_equal_count` 是严格数量控制：每张图保留与 Canny 完全相同数量的最大 Sobel 响应像素，只依赖 RGB 排序，不使用 GT。它是图像自适应的诊断对照，不等同于固定阈值部署。',
              '', '## 数量控制后的差异','', '以下为 Canny − 等数量 Sobel，单位百分点；按 validation 场景做 3000 次 paired bootstrap。']
    for metric,value in bootstrap['results'].items():
        lo,hi=value['interval_95_pp'];lines.append(f"- {metric}: {value['difference_pp']:+.3f} pp，95% 区间 [{lo:+.3f}, {hi:+.3f}]。")
    lines += ['', '## 图表','', '![密度与匹配曲线](figures/association_curves.png)','',
              '![主要对照](figures/primary_comparison.png)','', '![边界定义敏感性](figures/sensitivity.png)','',
              '## 解释范围','',
              '- Precision 表示 RGB 检出像素中，多少在 GT 边界 2 px 内；Recall 表示 GT 边界像素中，多少在 RGB 检出边缘 2 px 内。它们不是深度预测准确率。',
              '- 匹配是双向容差覆盖统计，不是一对一轮廓匹配，且 GT 边界本身保留跳变两端。',
              '- 随机参照为同一图、同一有效域、完全相同边缘数量的均匀无放回像素采样；统计使用其精确期望，非单次随机图片结果。',
              '- 同密度随机对照只能排除数量带来的基础覆盖机会，不能单独证明几何或语义因果关系。',
              '- GT 填充、RGB/深度对齐误差、光照与纹理均会影响匹配。未匹配 RGB 边缘不能全部标注为纹理；未匹配 GT 边界也不一定是真实物体轮廓。',
              '- Sobel 的连续强度作为网络输入与阈值化后的诊断结果是不同用途。Canny 匹配更好也不保证其训练收益更好。',
              '- 本实验没有调用 UniDepth，所以不提供原模型在四类区域内的预测误差；该项留待统一开发预测产生后补充。',
              '', '## 确定性选择的案例','']
    for e in examples:
        lines += [f"### {e['category']} / {e['sample_id']} / {e['scene']}",'',f"![{e['category']}]({e['file']})",'']
    (output/'analysis.md').write_text('\n'.join(lines),encoding='utf-8')
    cards=''.join(f'<h2>{html.escape(e["category"])} — {e["sample_id"]}</h2><img src="{e["file"]}" style="max-width:100%">' for e in examples)
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>RGB–Depth edge association</title><style>body{max-width:1400px;margin:30px auto;font:18px sans-serif;background:#f6f7f8;color:#222}img{background:white}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #aaa}</style><h1>RGB—深度边缘关联诊断</h1><p>577 fitting / 218 validation；未读取 test 图像。完整解释见 <a href="analysis.md">analysis.md</a>。</p><img width="100%" src="figures/association_curves.png"><img width="100%" src="figures/primary_comparison.png"><img width="100%" src="figures/sensitivity.png">'+cards,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=ROOT/'data/nyuv2')
    parser.add_argument('--split',type=Path,default=ROOT/'configs/edge_association_split.json')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/part3_improvement/edge_association/nyuv2_sobel_canny_20261118')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Use a fresh output path')
    config=json.loads(args.split.read_text(encoding='utf-8'))
    fits,vals=validate_split(config,args.data_root)
    args.output.mkdir(parents=True)
    cv2.setNumThreads(1)
    start=time.perf_counter()
    files=[Path(__file__),ROOT/'src/p3_depth/evaluation/rgb_depth_edges.py',ROOT/'src/p3_depth/temporal/fusion.py']
    metadata=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat(),
                  task='RGB/GT association only; no training, depth inference or test-image reads',
                  split_file_sha256=sha(args.split),split=config,settings=dict(targets=TARGETS,jumps=JUMPS,tolerances=TOLERANCES,primary_target=.05,primary_jump=.1,primary_tolerance=2),
                  implementation_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},
                  versions=dict(python=platform.python_version(),opencv=cv2.__version__,numpy=np.__version__),
                  git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  working_tree_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True))
    write_json(args.output/'metadata.json',metadata)
    with NYUv2Labeled(args.data_root/'nyu_depth_v2_labeled.mat') as dataset:
        settings,fit_scene_map=calibrate(dataset,fits,args.output)
        rows,fit_scenes=analyze(dataset,fits,'fitting',settings,set(fit_scene_map.values()))
        val_rows,val_scenes=analyze(dataset,vals,'validation',settings,set(fit_scenes))
        rows+=val_rows
        if set(fit_scenes)&set(val_scenes):raise ValueError('Scene overlap')
        write_csv(args.output/'per_image.csv',rows)
        summaries=summarize(rows)
        write_csv(args.output/'summary.csv',summaries);write_json(args.output/'summary.json',summaries)
        bootstrap=paired_bootstrap(rows);write_json(args.output/'paired_scene_bootstrap.json',bootstrap)
        make_plots(summaries,args.output)
        examples=make_examples(dataset,rows,settings,args.output)
    report(summaries,bootstrap,examples,args.output)
    metadata.update(status='complete',elapsed_seconds=time.perf_counter()-start,
                    images_read={'fitting':len(fits),'validation':len(vals),'test':0},
                    scene_counts={'fitting':len(fit_scenes),'validation':len(val_scenes)},
                    rows=len(rows),artifacts={str(p.relative_to(args.output)):sha(p) for p in sorted(args.output.rglob('*')) if p.is_file() and p.name!='metadata.json'})
    write_json(args.output/'metadata.json',metadata)
    print(json.dumps({'status':'PASS','images':795,'test_images_read':0,'output':str(args.output),'seconds':metadata['elapsed_seconds']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
