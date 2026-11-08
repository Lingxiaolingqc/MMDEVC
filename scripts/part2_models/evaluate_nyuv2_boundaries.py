"""Score saved NYUv2 depth discontinuities with fixed, explicit project thresholds."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import html
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from _common import ROOT, repo_path, read_yaml, write_json, write_csv, checksum
from p3_depth.data.nyuv2 import evaluation_mask
from p3_depth.evaluation.benchmark import select_nyu
from p3_depth.evaluation.depth import prepare_prediction,RelativeAlignmentError
from p3_depth.evaluation.boundary import evaluate_boundary
from p3_depth.evaluation.boundary import boundary_masks
from p3_depth.evaluation.course import score_course_nyu, PROTOCOL
from _paths import output_path,choose_output

SETTINGS={'relative_jump':.1,'tolerance_px':2.,'distance_cap_px':10.}


def prepare_boundary_prediction(raw,valid,gt,asset,protocol):
    region=evaluation_mask(gt)
    if protocol=='course_clipped':
        if (asset['prediction_type'],asset['unit'],asset['alignment'])!=('metric_depth','m','none'):
            raise ValueError('Course boundary analysis requires raw unaligned metric meters')
        score_course_nyu(raw,gt)  # Same shape, fixed GT domain, finite and clipping contract.
        depth=np.clip(raw.astype(np.float64),.001,10)
        return depth,np.isfinite(depth),region
    if protocol!='legacy':raise ValueError('Unknown boundary protocol')
    try:
        depth,valid,_=prepare_prediction(raw,gt,region,asset['prediction_type'],asset['unit'],asset['alignment'],valid)
    except RelativeAlignmentError:
        depth,valid=np.full_like(gt,np.nan),np.zeros_like(region)
    return depth,valid,region


def save_boundary_preview(rgb,depth,gt,region,valid,path,sid):
    _,_,gt_edges,pred_edges=boundary_masks(depth,gt,region,valid,SETTINGS['relative_jump'])
    h,w=rgb.shape[:2];canvas=Image.new('RGB',(3*w,h+48),(24,24,24));draw=ImageDraw.Draw(canvas)
    try:font=ImageFont.load_default(size=18)
    except TypeError:font=ImageFont.load_default()
    for i,(mask,label,color) in enumerate([(None,'RGB',None),(gt_edges,'GT edges (green)',(30,255,90)),
                                          (pred_edges,'Predicted edges (red)',(255,60,60))]):
        panel=rgb.copy()
        if mask is not None:panel[mask]=color
        canvas.paste(Image.fromarray(panel),(i*w,48));draw.text((i*w+8,5),sid+' | '+label,fill='white',font=font)
        draw.text((i*w+8,27),'10% depth jump | 2px matching tolerance',fill='white',font=font)
    canvas.save(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run')
    p.add_argument('--protocol',choices=('legacy','course_clipped'),default='legacy',
                   help='Legacy preserves previous behavior. Use course_clipped for the current raw metric run.')
    p.add_argument('--preview-count',type=int,default=6)
    p.add_argument('--paths-config',default='configs/paths.local.yaml')
    p.add_argument('--output-dir')
    a=p.parse_args()
    if a.preview_count<1:p.error('--preview-count must be positive')
    run=repo_path(a.run) if a.run else output_path('nyu',a.paths_config,model='unidepth',subset='test')
    meta=json.loads((run/'metadata.json').read_text(encoding='utf-8'))
    if meta['status']!='complete':
        raise ValueError('A completed image benchmark is required')
    identity=meta['identity'];cfg=identity['config'];asset=meta['model']['asset']
    output=choose_output(a.output_dir,'nyu_boundaries',a.paths_config,model=meta['model']['model'],subset=cfg['subset'],protocol=a.protocol)
    if output.exists():raise FileExistsError('Use a new boundary directory')
    if a.protocol=='course_clipped' and (meta['scope']!='standard_test_654' or meta['model']['gt_used_by_model']):
        raise ValueError('Current course analysis requires all 654 RGB-only test predictions')
    manifest,samples=select_nyu(ROOT,cfg['subset'],cfg['limit'])
    if a.protocol=='course_clipped' and len(samples)!=654:raise ValueError('Need all 654 course test images')
    if checksum(manifest)!=identity['manifest_sha256'] or [r['sample_id'] for r in samples]!=identity['sample_ids']:
        raise ValueError('Benchmark split has changed')
    root=repo_path(read_yaml(repo_path(a.paths_config))['data_roots']['nyuv2'])
    rows=[]
    for sample in samples:
        sid=sample['sample_id'];record=json.loads((run/'records'/f'{sid}.json').read_text(encoding='utf-8'))
        pp,mp,gp=run/'predictions'/f'{sid}.npy',run/'masks'/f'{sid}.npy',(root/sample['depth_gt_path']).resolve()
        if not gp.is_relative_to(root):
            raise ValueError('GT path escapes data root')
        for path,key in [(pp,'prediction_sha256'),(mp,'mask_sha256'),(gp,'gt_sha256')]:
            if checksum(path)!=record[key]:
                raise ValueError('Saved prediction/GT hash mismatch')
        raw,valid,gt=(np.load(path,allow_pickle=False) for path in (pp,mp,gp))
        depth,valid,region=prepare_boundary_prediction(raw,valid,gt,asset,a.protocol)
        rows.append({'sample_id':sid,**evaluate_boundary(depth,gt,region,valid,**SETTINGS),
                     'prediction_sha256':record['prediction_sha256'],'gt_sha256':record['gt_sha256'],
                     'record_sha256':checksum(run/'records'/f'{sid}.json')})
    ng=sum(r['gt_edge_pixels'] for r in rows);npred=sum(r['prediction_edge_pixels'] for r in rows)
    recall=sum(r['matched_gt_edge_pixels'] for r in rows)/ng if ng else None
    precision=sum(r['matched_prediction_edge_pixels'] for r in rows)/npred if npred else 0.
    summary={'images':len(rows),'gt_edge_pixels':ng,'prediction_edge_pixels':npred,
        'pooled_precision':precision,'pooled_recall':recall,
        'pooled_f1':2*precision*recall/(precision+recall) if recall is not None and precision+recall else 0.,
        'gt_edge_weighted_capped_distance_px':sum(r['gt_to_prediction_capped_distance_px']*r['gt_edge_pixels'] for r in rows if r['gt_edge_pixels'])/ng if ng else None,
        'images_without_gt_edges':sum(r['gt_edge_pixels']==0 for r in rows),
        'macro_f1':float(np.mean([r['f1'] for r in rows if r['f1'] is not None])) if any(r['f1'] is not None for r in rows) else None}
    output.mkdir(parents=True)
    write_csv(output/'per_image.csv',rows);write_json(output/'summary.json',summary)
    visual=output/'visualizations';visual.mkdir()
    ranked=sorted([r for r in rows if r['f1'] is not None],key=lambda r:(r['f1'],r['sample_id']))
    # Half worst-F1, half around median; deterministic selection, not a claim of representativeness.
    count=min(a.preview_count,len(ranked));low=(count+1)//2
    chosen=ranked[:low]
    for r in ranked[len(ranked)//2:]+ranked:
        if len(chosen)>=count:break
        if r not in chosen:chosen.append(r)
    previews=[]
    for row in chosen:
        sid=row['sample_id'];sample=next(s for s in samples if s['sample_id']==sid)
        rgb_path=(root/sample['rgb_path']).resolve()
        if not rgb_path.is_relative_to(root):raise ValueError('RGB path escapes data root')
        record=json.loads((run/'records'/f'{sid}.json').read_text(encoding='utf-8'))
        if checksum(rgb_path)!=record['rgb_sha256']:raise ValueError('RGB source changed')
        raw=np.load(run/'predictions'/f'{sid}.npy',allow_pickle=False)
        valid=np.load(run/'masks'/f'{sid}.npy',allow_pickle=False)
        gt=np.load(root/sample['depth_gt_path'],allow_pickle=False)
        depth,valid,region=prepare_boundary_prediction(raw,valid,gt,asset,a.protocol)
        with Image.open(rgb_path) as image:rgb=np.asarray(image.convert('RGB'))
        save_boundary_preview(rgb,depth,gt,region,valid,visual/f'{sid}.png',sid)
        previews.append({'sample_id':sid,'f1':row['f1'],'file':sid+'.png','sha256':checksum(visual/f'{sid}.png')})
    (visual/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Depth boundary diagnostics</h1>'+''.join(
        f'<p>{html.escape(e["sample_id"])} | F1={e["f1"]:.4f}</p><a href="{e["file"]}"><img width="1200" src="{e["file"]}"></a>' for e in previews),encoding='utf-8')
    write_json(output/'metadata.json',{'status':'complete','scope':meta['scope'],'source_run':str(run),
        'source_metadata_sha256':checksum(run/'metadata.json'),'manifest_sha256':identity['manifest_sha256'],
        'protocol':a.protocol,'depth_preparation':PROTOCOL if a.protocol=='course_clipped' else 'Historical prepare_prediction with the source asset alignment',
        'edge_aggregation':'Pooled precision/recall/F1 from edge counts; also unweighted macro F1 with undefined images excluded. Separate from the three required course accuracy metrics.',
        'raw_predictions_modified':False,'preview_selection':'Lowest F1 and median-F1 examples; deterministic, not random',
        'previews':previews,'preview_index_sha256':checksum(visual/'index.html'),
        'settings':SETTINGS,'definition':'max four-neighbor abs log-depth difference > log(1.1); both endpoints; 3x3 eroded fixed GT domain; Euclidean 2px match',
        'interpretation':'project depth-discontinuity metric, not a claim of reproducing a published boundary benchmark',
        'per_image_sha256':checksum(output/'per_image.csv'),'summary_sha256':checksum(output/'summary.json'),
        'implementation_sha256':{name:checksum(ROOT/name) for name in ['scripts/part2_models/evaluate_nyuv2_boundaries.py','src/p3_depth/evaluation/boundary.py','src/p3_depth/temporal/fusion.py','src/p3_depth/evaluation/depth.py']}})
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
