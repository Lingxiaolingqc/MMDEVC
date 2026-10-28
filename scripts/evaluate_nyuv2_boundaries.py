"""Score saved NYUv2 depth discontinuities with fixed, explicit project thresholds."""
import argparse
import json
import numpy as np

from run_stereo_baseline import ROOT,repo_path,read_yaml,write_json,write_csv,checksum
from p3_depth.data.nyuv2 import evaluation_mask
from p3_depth.evaluation.benchmark import select_nyu
from p3_depth.evaluation.depth import prepare_prediction,RelativeAlignmentError
from p3_depth.evaluation.boundary import evaluate_boundary

SETTINGS={'relative_jump':.1,'tolerance_px':2.,'distance_cap_px':10.}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True)
    p.add_argument('--paths-config',default='configs/paths.local.yaml')
    p.add_argument('--output-dir',required=True)
    a=p.parse_args()
    run,output=repo_path(a.run),repo_path(a.output_dir)
    if output.exists():
        raise FileExistsError('Use a new boundary directory')
    meta=json.loads((run/'metadata.json').read_text(encoding='utf-8'))
    if meta['status']!='complete':
        raise ValueError('A completed image benchmark is required')
    identity=meta['identity'];cfg=identity['config'];asset=meta['model']['asset']
    manifest,samples=select_nyu(ROOT,cfg['subset'],cfg['limit'])
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
        region=evaluation_mask(gt)
        try:
            depth,valid,_=prepare_prediction(raw,gt,region,asset['prediction_type'],asset['unit'],asset['alignment'],valid)
        except RelativeAlignmentError:
            depth,valid=np.full_like(gt,np.nan),np.zeros_like(region)
        rows.append({'sample_id':sid,**evaluate_boundary(depth,gt,region,valid,**SETTINGS)})
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
    write_json(output/'metadata.json',{'status':'complete','scope':meta['scope'],'source_run':a.run,
        'source_metadata_sha256':checksum(run/'metadata.json'),'manifest_sha256':identity['manifest_sha256'],
        'settings':SETTINGS,'definition':'max four-neighbor abs log-depth difference > log(1.1); both endpoints; 3x3 eroded fixed GT domain; Euclidean 2px match',
        'interpretation':'project depth-discontinuity metric, not a claim of reproducing a published boundary benchmark',
        'per_image_sha256':checksum(output/'per_image.csv'),'summary_sha256':checksum(output/'summary.json'),
        'implementation_sha256':{name:checksum(ROOT/name) for name in ['scripts/evaluate_nyuv2_boundaries.py','src/p3_depth/evaluation/boundary.py','src/p3_depth/temporal/fusion.py','src/p3_depth/evaluation/depth.py']}})
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
