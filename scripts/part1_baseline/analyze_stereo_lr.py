"""Analyze saved custom-LR filtering without rerunning either SGBM matcher."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import numpy as np
from _common import ROOT, repo_path, read_yaml, checksum, select_sample, write_csv, write_json
from _paths import choose_output, output_path
from p3_depth.data.middlebury import EXPECTED_SCENES, load_sample, disparity_to_depth_m
from p3_depth.models.stereo_sgbm import left_right_consistency
from p3_depth.evaluation.stereo import evaluate_region, compare_lr_rejection


def preview(sample, before, after, removed, destination, upper):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#b4b4b4')
    axes[0,0].imshow(sample.left_rgb);axes[0,0].set_title('RGB')
    for ax,values,title in [(axes[0,1],np.where(sample.valid_gt,sample.disparity_px,np.nan),'GT disparity'),
                            (axes[0,2],before,'Before custom LR'),(axes[1,0],after,'After custom LR')]:
        im=ax.imshow(np.ma.masked_invalid(values),cmap=cmap,vmin=0,vmax=upper)
        ax.set_title(title);fig.colorbar(im,ax=ax,label='px',shrink=.7)
    axes[1,1].imshow(removed,cmap='gray',vmin=0,vmax=1)
    axes[1,1].set_title('Newly rejected: white')
    labels=np.zeros((*removed.shape,3),np.uint8)
    error=np.abs(before.astype(float)-sample.disparity_px)
    labels[removed & ~sample.valid_gt]=(180,180,180)
    labels[removed & sample.valid_gt & (error>1)]=(40,210,90)
    labels[removed & sample.valid_gt & (error<=1)]=(245,70,70)
    axes[1,2].imshow(labels)
    axes[1,2].set_title('Rejected: green >1px; red <=1px; gray no GT')
    for ax in axes.flat:ax.axis('off')
    fig.suptitle(sample.paths.sample_id+' | OpenCV internal settings unchanged | custom filter only')
    fig.savefig(destination,dpi=140);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',help='Completed 15-scene suite; default current Part 1/all')
    p.add_argument('--manifest',default='manifests/middlebury.csv')
    p.add_argument('--paths-config',default='configs/paths.local.yaml')
    p.add_argument('--output-dir')
    a=p.parse_args()
    run=repo_path(a.run) if a.run else output_path('stereo_suite',a.paths_config,phase='all')
    out=choose_output(a.output_dir,'stereo_lr',a.paths_config)
    if out.exists():raise FileExistsError('Use a new LR analysis directory; sources are preserved')
    manifest=repo_path(a.manifest)
    root=repo_path(read_yaml(repo_path(a.paths_config))['data_roots']['middlebury'])
    # Validate every source before creating an output folder.
    sources=[]
    for sid in EXPECTED_SCENES:
        folder=run/sid;meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
        if meta['status']!='completed' or meta['manifest_sha256']!=checksum(manifest):
            raise ValueError('Incomplete stereo source or changed manifest: '+sid)
        for source in meta['sources']:
            if checksum(root/source['path'])!=source['sha256']:raise ValueError('Stereo data changed: '+sid)
        keys=['diagnostics/left_before_lr_px.npy','diagnostics/right_signed_px.npy','diagnostics/lr_error_px.npy',
              f'disparities/{sid}.npy',f'masks/{sid}.npy','config_resolved.yaml']
        for key in keys:
            if checksum(folder/key)!=meta['artifacts_sha256'][key]:raise ValueError('Stereo prediction changed: '+sid)
        sources.append({'sample_id':sid,'metadata_sha256':checksum(folder/'metadata.json')})
    out.mkdir(parents=True);(out/'visualizations').mkdir();(out/'rejected_masks').mkdir()
    rows=[];rejections=[]
    for sid in EXPECTED_SCENES:
        folder=run/sid
        paths,split=select_sample(manifest,root,sid);sample=load_sample(paths)
        before=np.load(folder/'diagnostics/left_before_lr_px.npy',allow_pickle=False)
        right=np.load(folder/'diagnostics/right_signed_px.npy',allow_pickle=False)
        after=np.load(folder/'disparities'/f'{sid}.npy',allow_pickle=False)
        valid=np.load(folder/'masks'/f'{sid}.npy',allow_pickle=False)
        cfg=read_yaml(folder/'config_resolved.yaml')
        passed,residual=left_right_consistency(before,right,cfg['lr_max_diff_px'])
        before_depth=disparity_to_depth_m(before,sample.calibration)
        before_valid=np.isfinite(before)&np.isfinite(before_depth)&(before_depth>0)
        if not np.array_equal(residual,np.load(folder/'diagnostics/lr_error_px.npy'),equal_nan=True):
            raise ValueError('Saved LR residual does not recompute: '+sid)
        if not np.array_equal(valid,passed&before_valid):raise ValueError('Saved final LR mask does not recompute: '+sid)
        if not np.array_equal(after,np.where(valid,before,np.nan),equal_nan=True):
            raise ValueError('Custom LR must only filter, not modify disparities: '+sid)
        removed=before_valid&~valid
        for region_name,region in [('all_gt',sample.valid_gt),('nonoccluded_gt',sample.valid_gt&sample.nonoccluded)]:
            for state,disp,depth,mask in [('off',before,before_depth,before_valid),
                                          ('on',after,disparity_to_depth_m(after,sample.calibration),valid)]:
                score=evaluate_region(depth,disp,mask,sample.depth_m,sample.disparity_px,region)
                rows.append({'sample_id':sid,'split':split,'region':region_name,'custom_lr':state,
                             **{k:v for k,v in score.items() if k!='depth_on_valid_predictions'}})
            rejections.append({'sample_id':sid,'split':split,'region':region_name,
                               **compare_lr_rejection(before,after,before_valid,valid,sample.disparity_px,region),
                               'removed_no_comparable_right':int((removed&region&~np.isfinite(residual)).sum()),
                               'removed_residual_above_threshold':int((removed&region&np.isfinite(residual)).sum())})
        np.save(out/'rejected_masks'/f'{sid}.npy',removed)
        preview(sample,np.where(before_valid,before,np.nan),after,removed,out/'visualizations'/f'{sid}.png',cfg['num_disparities'])
    summary=[]
    for split in ('all_15_descriptive','dev','evaluation'):
        for region in ('all_gt','nonoccluded_gt'):
            for state in ('off','on'):
                selected=[r for r in rows if r['region']==region and r['custom_lr']==state and (split=='all_15_descriptive' or r['split']==split)]
                n=sum(r['gt_pixels'] for r in selected);k=sum(r['valid_prediction_pixels'] for r in selected)
                errors=[r['disparity_mae_px_on_valid'] for r in selected if r['disparity_mae_px_on_valid'] is not None]
                summary.append({'split':split,'region':region,'custom_lr':state,'scenes':len(selected),
                                'gt_pixels':n,'valid_prediction_pixels':k,'pooled_coverage':k/n if n else None,
                                'macro_disparity_mae_px_on_valid':float(np.mean(errors)) if errors else None,
                                'mae_contributing_scenes':len(errors)})
    write_csv(out/'per_scene.csv',rows);write_csv(out/'rejections.csv',rejections);write_csv(out/'summary.csv',summary)
    write_json(out/'metadata.json',{'status':'complete','kind':'posthoc_custom_lr_analysis','scenes':len(EXPECTED_SCENES),
        'source_run':str(run),'sources':sources,'manifest_sha256':checksum(manifest),'raw_predictions_modified':False,
        'comparison':'Same saved SGBM matches, calibration and physical-depth validity. Only the extra custom LR filter differs. OpenCV internal disp12MaxDiff remains unchanged.',
        'interpretation':'Conditional MAE must be read with coverage. Filtering cannot change retained values. Rejected >1px fraction uses rejected pixels with GT as denominator; <=1px is threshold agreement, not exact correctness.',
        'aggregation':'Equal scene mean MAE with contributing counts; pooled coverage uses total valid/GT pixels. All 15 includes development scenes and is descriptive.',
        'implementation_sha256':{name:checksum(ROOT/name) for name in ['scripts/part1_baseline/analyze_stereo_lr.py','src/p3_depth/evaluation/stereo.py','src/p3_depth/models/stereo_sgbm.py']},
        'files_sha256':{f.relative_to(out).as_posix():checksum(f) for f in sorted(out.rglob('*')) if f.is_file()}})
    print(f'PASS: {len(EXPECTED_SCENES)} saved stereo pairs analyzed, no matcher inference: {out}')


if __name__=='__main__':main()
