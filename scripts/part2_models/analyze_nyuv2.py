"""Audit saved course NYUv2 predictions and export error, scale and boundary evidence."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from _common import ROOT, repo_path, read_yaml, checksum, write_csv, write_json
from _video_review import read_json, read_rows
from render_depth_video import colorize
from p3_depth.data.nyuv2 import evaluation_mask, read_standard_split
from p3_depth.evaluation.benchmark import select_nyu
from p3_depth.evaluation.course import PROTOCOL, score_course_nyu, summarize_course
from p3_depth.evaluation.depth import depth_values_metrics
from p3_depth.evaluation.boundary import evaluate_boundary, boundary_masks


def same_values(actual, saved, label):
    for key, value in actual.items():
        if value is None:
            if saved[key] not in (None, ''):
                raise ValueError(f'{label}: unexpected {key}')
        elif not np.isclose(value, float(saved[key]), rtol=1e-10, atol=1e-12):
            raise ValueError(f'{label}: changed {key}')


def checked_provenance(folder, meta, run):
    if meta['status'] != 'complete' or meta['source_metadata_sha256'] != checksum(run/'metadata.json'):
        raise ValueError('Evaluation belongs to a different or incomplete inference run')
    for file, key in [('per_image.csv', 'per_image_sha256'), ('summary.json', 'summary_sha256')]:
        if checksum(folder/file) != meta[key]:
            raise ValueError('Saved evaluation changed: '+str(folder/file))
    for file, digest in meta['implementation_sha256'].items():
        if checksum(ROOT/file) != digest:
            raise ValueError('Evaluation implementation changed: '+file)


def distribution(values):
    q = np.quantile(values, [0, .05, .25, .5, .75, .95, .99, 1])
    return {**dict(zip(('min', 'p05', 'q1', 'median', 'q3', 'p95', 'p99', 'max'), map(float, q))),
            'mean': float(np.mean(values)), 'images': len(values)}


def figures(out, rows, bins, summary):
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.2), layout='constrained')
    for ax, key, title in zip(axes, ('abs_rel', 'rmse_m', 'delta1', 'boundary_f1'),
                              ('AbsRel (lower better)', 'RMSE, m (lower better)',
                               'Delta1 (higher better)', 'Boundary F1 (higher better)')):
        d = distribution([r[key] for r in rows])
        ax.bxp([dict(med=d['median'], q1=d['q1'], q3=d['q3'], whislo=d['p05'],
                     whishi=d['p95'], mean=d['mean'], fliers=[])], showmeans=True, showfliers=False)
        ax.set(title=title, xticks=[], ylabel='Per-image score')
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('654 test images | P5-P95 whiskers, quartile boxes, median lines, triangle means\n'
                 'Tails retained in CSV; correlated images are not independent trials')
    fig.savefig(out/'figures/per_image_distributions.png', dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), layout='constrained')
    axes[0].hist([r['median_prediction_gt_ratio'] for r in rows], bins=35, color='#3b7b9e')
    axes[0].axvline(1, color='black', linestyle='--')
    axes[0].set(title='Per-image median prediction / GT', xlabel='Ratio (1 = no median multiplicative bias)', ylabel='Images')
    axes[1].scatter([r['mean_gt_m'] for r in rows], [r['mean_signed_error_m'] for r in rows], s=9, alpha=.4)
    axes[1].axhline(0, color='black', linestyle='--')
    axes[1].set(title='Image depth and signed error', xlabel='Mean GT depth (m)', ylabel='Mean prediction - GT (m)')
    x = np.arange(len(bins))
    axes[2].bar(x-.18, [r['mae_m'] for r in bins], .36, label='MAE')
    axes[2].bar(x+.18, [r['rmse_m'] for r in bins], .36, label='RMSE')
    axes[2].set(title='Pixel-pooled errors by GT distance', xticks=x,
                xticklabels=[r['bin_m'] for r in bins], xlabel='GT depth interval (m)', ylabel='Error (m)')
    axes[2].legend()
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle('Course-clipped predictions | descriptive diagnostics, no GT fitting or correction\n'
                 'Distance groups mix scenes and objects; this is not a distance-only controlled experiment')
    fig.savefig(out/'figures/scale_and_distance.png', dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), layout='constrained')
    b = summary['boundary']
    axes[0].bar(['Precision', 'Recall', 'F1'], [b['pooled_precision'], b['pooled_recall'], b['pooled_f1']])
    axes[0].set(title='Pooled boundary scores', ylim=(0, 1), ylabel='Fraction')
    axes[1].scatter([r['abs_rel'] for r in rows], [r['boundary_f1'] for r in rows], s=9, alpha=.4)
    axes[1].set(title='Depth accuracy and sharp edges', xlabel='Course AbsRel', ylabel='Boundary F1')
    axes[2].scatter([r['gt_edge_pixels'] for r in rows], [r['prediction_edge_pixels'] for r in rows], s=9, alpha=.4)
    max_count=max(max(r['gt_edge_pixels'],r['prediction_edge_pixels']) for r in rows)
    axes[2].plot([0, max_count], [0, max_count], 'k--', linewidth=1)
    axes[2].set(title='Thresholded edge counts per image', xlabel='GT edge pixels', ylabel='Prediction edge pixels')
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle('Supplemental depth discontinuities | >10% adjacent depth jump, 2 px tolerance\n'
                 'Depth edges are not all RGB texture edges; F1 is not a whole-scene accuracy score')
    fig.savefig(out/'figures/boundaries.png', dpi=180); plt.close(fig)


def sample_figure(rgb, pred, gt, row, settings, path):
    region = evaluation_mask(gt)
    valid = np.isfinite(pred)
    _, _, ge, pe = boundary_masks(pred, gt, region, valid, settings['relative_jump'])
    overlay=rgb.copy(); overlay[ge]=(40,255,80); overlay[pe]=(255,30,200); overlay[ge & pe]=(255,255,255)
    err = np.ma.masked_where(~region, np.abs(pred-gt))
    bias = np.ma.masked_where(~region, pred-gt)
    depthmap=plt.get_cmap('viridis').copy(); depthmap.set_bad('#cccccc')
    errmap=plt.get_cmap('magma').copy(); errmap.set_bad('#cccccc')
    biasmap=plt.get_cmap('coolwarm').copy(); biasmap.set_bad('#cccccc')
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.8), layout='constrained')
    axes[0,0].imshow(rgb); axes[0,0].set_title('RGB | yellow box = course crop')
    from matplotlib.patches import Rectangle
    axes[0,0].add_patch(Rectangle((41,45),560,426,edgecolor='yellow',facecolor='none',lw=1))
    panels=[(axes[0,1],np.ma.masked_where(~np.isfinite(gt)|(gt<=0),gt),depthmap,0,10,'GT filled depth (m)'),
            (axes[0,2],pred,depthmap,0,10,'UniDepthV2 course-clipped (m)'),
            (axes[1,0],err,errmap,0,2,'Absolute error (m), display capped at 2'),
            (axes[1,1],bias,biasmap,-2,2,'Prediction - GT (m), display +/-2')]
    for ax, data, cmap, lo, hi, title in panels:
        im=ax.imshow(data,cmap=cmap,vmin=lo,vmax=hi); ax.set_title(title); fig.colorbar(im,ax=ax,shrink=.7)
    axes[1,2].imshow(overlay); axes[1,2].set_title('Edges: GT green, pred magenta, exact overlap white')
    for ax in axes.flat: ax.set_axis_off()
    fig.suptitle(f"{row['sample_id']} | {row['scene']} | AbsRel {row['abs_rel']:.4f}, RMSE {row['rmse_m']:.3f} m, "
                 f"delta1 {row['delta1']:.3f}, edge F1 {row['boundary_f1']:.3f}\n"
                 'Common depth/error ranges; error and edge evaluation only inside the fixed GT domain')
    fig.savefig(path,dpi=150); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', default='outputs/part2_models/unidepth/test')
    p.add_argument('--output-dir')
    p.add_argument('--paths-config', default='configs/paths.local.yaml')
    args=p.parse_args()
    run=repo_path(args.run); out=repo_path(args.output_dir) if args.output_dir else run/'analysis'
    if out.exists(): raise FileExistsError('Preserve prior analysis; use a new --output-dir')
    meta=read_json(run/'metadata.json'); asset=meta['model']['asset']
    if (meta['status'],meta['scope'],asset['prediction_type'],asset['unit'],asset['alignment'],meta['model']['gt_used_by_model']) != ('complete','standard_test_654','metric_depth','m','none',False):
        raise ValueError('Require completed 654-image RGB-only, unaligned metric predictions')
    course=run/'course_evaluation'; boundary=run/'boundary_course_clipped'
    cm=read_json(course/'provenance.json'); bm=read_json(boundary/'metadata.json')
    checked_provenance(course,cm,run); checked_provenance(boundary,bm,run)
    if cm['protocol'] != PROTOCOL or bm['protocol'] != 'course_clipped' or bm['depth_preparation'] != PROTOCOL:
        raise ValueError('Mismatched course/boundary protocol')
    manifest, samples=select_nyu(ROOT,'test')
    ids=[s['sample_id'] for s in samples]
    if ids!=meta['identity']['sample_ids'] or any(x!=checksum(manifest) for x in (cm['manifest_sha256'],bm['manifest_sha256'],meta['identity']['manifest_sha256'])):
        raise ValueError('Different manifest/split')
    data=repo_path(read_yaml(repo_path(args.paths_config))['data_roots']['nyuv2'])
    split=read_standard_split(data/'splits.mat')
    if {f'nyuv2_{i:04d}' for i in split['test']} != set(ids): raise ValueError('Not official test indices')
    course_rows=read_rows(course/'per_image.csv'); boundary_rows=read_rows(boundary/'per_image.csv')
    if len(course_rows)!=654 or len(boundary_rows)!=654: raise ValueError('Wrong evaluation row count')
    cs={r['sample_id']:r for r in course_rows}
    bs={r['sample_id']:r for r in boundary_rows}
    if set(cs)!=set(ids) or set(bs)!=set(ids): raise ValueError('Incomplete evaluation records')
    edges=np.array([.001,1,2,3,4,6,10.000001]); accum=np.zeros((6,8),np.float64)
    rows=[]; scored=[]; boundaries=[]; record_hashes={}; timings=[]
    for i,s in enumerate(samples):
        sid=s['sample_id']; record_path=run/'records'/f'{sid}.json'; record=read_json(record_path)
        paths=[run/'predictions'/f'{sid}.npy',run/'masks'/f'{sid}.npy',(data/s['depth_gt_path']).resolve(),(data/s['rgb_path']).resolve()]
        if not all(x.is_relative_to(data) for x in paths[2:]): raise ValueError('Data path escaped data root')
        if record['sample_id']!=sid or record['split']!='test': raise ValueError('Changed sample identity')
        for path,key in zip(paths,('prediction_sha256','mask_sha256','gt_sha256','rgb_sha256')):
            if checksum(path)!=record[key]: raise ValueError('Changed source: '+str(path))
        rh=checksum(record_path); record_hashes[sid]=rh
        if rh!=cs[sid]['source_record_sha256'] or rh!=bs[sid]['record_sha256']: raise ValueError('Changed source record')
        for key in ('prediction_sha256','gt_sha256'):
            if record[key]!=cs[sid][key] or record[key]!=bs[sid][key]: raise ValueError('Changed evaluated input')
        raw,valid,gt=(np.load(path,allow_pickle=False) for path in paths[:3]); region=evaluation_mask(gt)
        if valid.dtype!=np.bool_ or valid.shape!=raw.shape or not np.array_equal(valid,np.isfinite(raw)&(raw>0)):
            raise ValueError('Unexpected prediction mask')
        score=score_course_nyu(raw,gt); same_values(score,cs[sid],sid+' course'); scored.append({'sample_id':sid,**score})
        g=gt[region].astype(np.float64); r=raw[region].astype(np.float64); pred=np.clip(raw.astype(np.float64),.001,10)
        raw_scores=depth_values_metrics(r,g); same_values(raw_scores,record['metrics']['depth_on_valid_predictions'],sid+' raw')
        if score['abs_rel']>raw_scores['abs_rel']+1e-12 or score['rmse_m']>raw_scores['rmse_m']+1e-12 or score['delta1']<raw_scores['delta1']-1e-12:
            raise ValueError('Clipping to the valid GT interval unexpectedly worsened metrics')
        b=evaluate_boundary(pred,gt,region,np.isfinite(pred),**bm['settings']); same_values(b,bs[sid],sid+' boundary'); boundaries.append(b)
        z=pred[region]; e=z-g; ae=np.abs(e); ratio=z/g
        q=np.quantile(ae,[.05,.25,.5,.75,.95,.99])
        scene=re.search(r'scene=([^;]+)',s['notes'])
        row={'sample_id':sid,'scene':scene.group(1) if scene else 'unspecified',**score,
             'mae_m':float(ae.mean()),'mean_signed_error_m':float(e.mean()),'mean_gt_m':float(g.mean()),
             'median_prediction_gt_ratio':float(np.median(ratio)),'mean_log_prediction_gt_ratio':float(np.log(ratio).mean()),
             'raw_abs_rel':raw_scores['abs_rel'],'raw_rmse_m':raw_scores['rmse_m'],'raw_delta1':raw_scores['delta1'],
             'raw_max_m':float(r.max()),'prediction_coverage':float(valid[region].mean()),
             **{f'pixel_abs_error_{name}_m':float(v) for name,v in zip(('p05','q1','median','q3','p95','p99'),q)},
             'boundary_f1':b['f1'],'gt_edge_pixels':b['gt_edge_pixels'],'prediction_edge_pixels':b['prediction_edge_pixels']}
        rows.append(row); timings.append(record['inference_seconds'])
        for j,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
            m=(g>=lo)&(g<hi); n=int(m.sum())
            if n: accum[j]+=[n,1,ae[m].sum(),(e[m]**2).sum(),(ae[m]/g[m]).sum(),e[m].sum(),ratio[m].sum(),(np.maximum(ratio[m],1/ratio[m])<1.25).sum()]
        if (i+1)%100==0: print(f'Checked {i+1}/654',flush=True)
    summary=summarize_course(scored)
    same_values(summary,read_json(course/'summary.json'),'course summary')
    ng=sum(b['gt_edge_pixels'] for b in boundaries); npred=sum(b['prediction_edge_pixels'] for b in boundaries)
    precision=sum(b['matched_prediction_edge_pixels'] for b in boundaries)/npred
    recall=sum(b['matched_gt_edge_pixels'] for b in boundaries)/ng
    boundary_summary={'images':len(rows),'gt_edge_pixels':ng,'prediction_edge_pixels':npred,
        'pooled_precision':precision,'pooled_recall':recall,'pooled_f1':2*precision*recall/(precision+recall),
        'gt_edge_weighted_capped_distance_px':sum(b['gt_to_prediction_capped_distance_px']*b['gt_edge_pixels'] for b in boundaries)/ng,
        'images_without_gt_edges':sum(b['gt_edge_pixels']==0 for b in boundaries),'macro_f1':float(np.mean([b['f1'] for b in boundaries]))}
    same_values(boundary_summary,read_json(boundary/'summary.json'),'boundary summary')
    bins=[]
    for j,a in enumerate(accum):
        n,images,ab,sq,rel,signed,ratio,d1=a
        bins.append({'bin_m':f'{edges[j]:g}-{min(edges[j+1],10):g}', 'pixels':int(n),'images':int(images),
                     'fraction_all_gt':float(n/summary['gt_pixels']), 'mae_m':float(ab/n) if n else None,
                     'rmse_m':float(np.sqrt(sq/n)) if n else None, 'abs_rel':float(rel/n) if n else None,
                     'mean_signed_error_m':float(signed/n) if n else None,'mean_prediction_gt_ratio':float(ratio/n) if n else None,
                     'delta1':float(d1/n) if n else None})
    if sum(b['pixels'] for b in bins)!=summary['gt_pixels']: raise ValueError('Distance groups lost GT pixels')
    ranked=sorted(rows,key=lambda r:(r['abs_rel'],r['sample_id']))
    chosen={ranked[round(q*(len(rows)-1))]['sample_id']:f'AbsRel rank quantile {q:g}' for q in (0,.25,.5,.75,.99,1)}
    for r in (min(rows,key=lambda r:r['median_prediction_gt_ratio']),max(rows,key=lambda r:r['median_prediction_gt_ratio'])):
        chosen.setdefault(r['sample_id'],'Extreme median prediction/GT ratio')
    for item in bm['previews']: chosen.setdefault(item['sample_id'],'Existing lowest/median boundary F1 selection')
    result={'course':summary,'boundary':boundary_summary,
        'images_with_upper_clipping':sum(r['clipped_high_pixels']>0 for r in rows),
        'upper_clip_fraction':summary['clipped_high_pixels']/summary['gt_pixels'],
        'zero_prediction_edge_images':sum(b['prediction_edge_pixels']==0 for b in boundaries),
        'zero_boundary_f1_images':sum(b['f1']==0 for b in boundaries),
        'scene_count':len({r['scene'] for r in rows}),
        'minimum_prediction_coverage':min(r['prediction_coverage'] for r in rows),
        'raw_unclipped_macro':{k:float(np.mean([r['raw_'+k] for r in rows])) for k in ('abs_rel','rmse_m','delta1')},
        'pixel_pooled_rmse_m':float(np.sqrt(accum[:,3].sum()/summary['gt_pixels'])),
        'per_image_distributions':{k:distribution([r[k] for r in rows]) for k in ('abs_rel','rmse_m','delta1','boundary_f1','median_prediction_gt_ratio','mean_signed_error_m')},
        'selection':chosen}
    resources={k:meta[k] for k in ('model','timing','warmup','peak_cuda_allocated_bytes','peak_cuda_reserved_bytes')}
    resources.update(images=len(rows),inference_seconds=sum(timings),mean_seconds_per_image=float(np.mean(timings)),fps=len(rows)/sum(timings))
    out.mkdir(parents=True); (out/'figures').mkdir(); (out/'samples').mkdir()
    write_csv(out/'per_image.csv',rows); write_csv(out/'depth_bins.csv',bins)
    write_json(out/'summary.json',result); write_json(out/'resources.json',resources)
    figures(out,rows,bins,result)
    for s in samples:
        sid=s['sample_id']
        if sid not in chosen: continue
        with Image.open(data/s['rgb_path']) as image: rgb=np.array(image.convert('RGB'))
        gt=np.load(data/s['depth_gt_path'],allow_pickle=False)
        pred=np.clip(np.load(run/'predictions'/f'{sid}.npy',allow_pickle=False).astype(np.float64),.001,10)
        row=next(r for r in rows if r['sample_id']==sid)
        sample_figure(rgb,pred,gt,row,bm['settings'],out/'samples'/f'{sid}.png')
    for start in range(0,len(chosen),4):
        ids_page=list(chosen)[start:start+4]; sheet=Image.new('RGB',(1280,len(ids_page)*270),'white'); draw=ImageDraw.Draw(sheet)
        font=ImageFont.load_default(size=16)
        for j,sid in enumerate(ids_page):
            s=next(s for s in samples if s['sample_id']==sid)
            with Image.open(data/s['rgb_path']) as im: rgb=np.array(im.convert('RGB'))
            gt=np.load(data/s['depth_gt_path'],allow_pickle=False); region=evaluation_mask(gt)
            pred=np.clip(np.load(run/'predictions'/f'{sid}.npy',allow_pickle=False),.001,10)
            panels=[rgb,colorize(gt,np.isfinite(gt)&(gt>0),0,10),colorize(pred,np.isfinite(pred),0,10),
                    colorize(np.abs(pred.astype(np.float64)-gt),region,0,2)]
            for k,(array,label) in enumerate(zip(panels,('RGB','GT 0-10m','Pred 0-10m','Abs error 0-2m'))):
                sheet.paste(Image.fromarray(array).resize((320,240),Image.Resampling.LANCZOS),(k*320,j*270+30))
                draw.text((k*320+4,j*270+5),sid+' | '+label,font=font,fill='black')
        sheet.save(out/'figures'/f'overview_{start//4+1}.png')
    source_files=(run/'metadata.json',course/'provenance.json',course/'summary.json',course/'per_image.csv',
                  boundary/'metadata.json',boundary/'summary.json',boundary/'per_image.csv',manifest,data/'splits.mat')
    implementation=('scripts/part2_models/analyze_nyuv2.py','src/p3_depth/evaluation/course.py',
                    'src/p3_depth/evaluation/boundary.py','src/p3_depth/evaluation/depth.py','src/p3_depth/data/nyuv2.py')
    write_json(out/'quantitative_review.json',{'status':'quantitative_complete_visual_pending','images':len(rows),
        'source_sha256':{str(x.relative_to(ROOT)) if x.is_relative_to(ROOT) else str(x):checksum(x) for x in source_files},
        'record_sha256':record_hashes,'implementation_sha256':{x:checksum(ROOT/x) for x in implementation},
        'selection':chosen,'protocol':PROTOCOL,'boundary_settings':bm['settings'],
        'scope':'All 654 course and boundary scores recomputed; predictions, masks, RGB, GT and records hash-checked. No model inference, GT alignment or prediction changes.',
        'limitations':'Ratio is diagnostic only, not a fitted scale. Depth bins pool pixels and mix scenes/objects. Image distributions are descriptive, not independent-trial significance estimates. Preview selection is post-hoc diagnostic, not random.',
        'artifacts_sha256':{x.relative_to(out).as_posix():checksum(x) for x in out.rglob('*') if x.is_file()}})
    print({'images':len(rows),'selected_images':len(chosen),'output':str(out)},flush=True)


if __name__=='__main__': main()
