"""Verify the revised positive-disparity GT rule against the saved 15-pair baseline."""
import argparse
import csv
import json
import numpy as np
from run_stereo_baseline import ROOT,repo_path,read_yaml,checksum,write_csv,write_json
from p3_depth.data.middlebury import discover_samples,load_sample,EXPECTED_SCENES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='outputs/stereo_all_week1_20261021')
    p.add_argument('--output-dir',required=True)
    a=p.parse_args();run,output=repo_path(a.run),repo_path(a.output_dir)
    if output.exists():raise FileExistsError('Use a new audit directory')
    data=repo_path(read_yaml(ROOT/'configs/paths.local.yaml')['data_roots']['middlebury'])
    samples=discover_samples(data)
    if {x.sample_id for x in samples}!=set(EXPECTED_SCENES):raise ValueError('Need all 15 trainingQ pairs')
    with (ROOT/'results/tables/stereo_week1_per_scene.csv').open(encoding='utf-8',newline='') as f:
        table={r['sample_id']:r for r in csv.DictReader(f) if r['region']=='all_gt'}
    rows=[]
    for paths in samples:
        s=load_sample(paths);name=paths.sample_id;folder=run/name
        meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
        for item in meta['sources']:
            if checksum(data/item['path'])!=item['sha256']:raise ValueError('Stereo input changed')
        for key in ('disparities/'+name+'.npy','masks/'+name+'.npy','diagnostics/evaluation_all_gt.npy'):
            if checksum(folder/key)!=meta['artifacts_sha256'][key]:raise ValueError('Stereo artifact changed')
        gt=np.isfinite(s.disparity_px)&(s.disparity_px>0)
        if not np.array_equal(gt,s.valid_gt) or not np.array_equal(gt,np.load(folder/'diagnostics/evaluation_all_gt.npy')):
            raise ValueError('Existing GT mask differs from revised finite-positive rule')
        pred=np.load(folder/'disparities'/f'{name}.npy');valid=np.load(folder/'masks'/f'{name}.npy')&gt
        n,k=int(gt.sum()),int(valid.sum());mae=float(np.abs(pred[valid].astype(float)-s.disparity_px[valid]).mean())
        old=table[name]
        if n!=int(old['gt_pixels']) or k!=int(old['valid_prediction_pixels']) or not np.isclose(mae,float(old['disparity_mae_px_on_valid']),rtol=0,atol=1e-10):
            raise ValueError('Recomputed course stereo score differs')
        rows.append({'sample_id':name,'gt_pixels':n,'valid_prediction_pixels':k,'valid_fraction':k/n,
                     'disparity_mae_px':mae,'finite_nonpositive_gt_pixels':int((np.isfinite(s.disparity_px)&(s.disparity_px<=0)).sum()),
                     'source_metadata_sha256':checksum(folder/'metadata.json')})
    output.mkdir(parents=True);write_csv(output/'per_scene.csv',rows)
    write_json(output/'audit.json',{'status':'verified','scenes':len(rows),'gt_rule':'finite disparity > 0',
        'existing_gt_masks_match':True,'source_arrays_modified':False,'pooled_valid_fraction':sum(r['valid_prediction_pixels'] for r in rows)/sum(r['gt_pixels'] for r in rows),
        'macro_disparity_mae_px':float(np.mean([r['disparity_mae_px'] for r in rows])),
        'per_scene_sha256':checksum(output/'per_scene.csv'),'implementation_sha256':checksum(ROOT/'scripts/audit_course_stereo.py')})
    print(f'PASS: all {len(rows)} GT masks and saved disparity MAE/valid fractions agree')


if __name__=='__main__':main()
