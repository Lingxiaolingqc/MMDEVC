"""Export full-test tables, provenance, and deterministically selected qualitative cases."""
import argparse
import csv
import json
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from run_stereo_baseline import ROOT,repo_path,read_yaml,write_json,write_csv,checksum
from p3_depth.data.nyuv2 import evaluation_mask
from p3_depth.evaluation.depth import prepare_prediction,RelativeAlignmentError


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,action='append')
    p.add_argument('--boundary',required=True,action='append',help='One boundary folder per run, in the same order')
    p.add_argument('--paths-config',default='configs/paths.local.yaml')
    p.add_argument('--output-dir',required=True)
    a=p.parse_args()
    if len(a.run)!=len(a.boundary):
        p.error('Each run needs one corresponding boundary result')
    output=repo_path(a.output_dir)
    if output.exists():
        raise FileExistsError('Use a new summary directory')
    runs=[];table=[];provenance=[]
    for source,bsource in zip(a.run,a.boundary):
        run,bdir=repo_path(source),repo_path(bsource)
        meta=json.loads((run/'metadata.json').read_text(encoding='utf-8'))
        scores=json.loads((run/'summary.json').read_text(encoding='utf-8'))
        bm=json.loads((bdir/'metadata.json').read_text(encoding='utf-8'))
        boundary=json.loads((bdir/'summary.json').read_text(encoding='utf-8'))
        if meta['status']!='complete' or meta['scope']!='standard_test_654' or scores['images']!=654:
            raise ValueError('Only complete 654-image standard tests belong in this export')
        if bm['status']!='complete' or bm['source_metadata_sha256']!=checksum(run/'metadata.json'):
            raise ValueError('Boundary results do not correspond to this image run')
        if checksum(bdir/'summary.json')!=bm['summary_sha256'] or checksum(bdir/'per_image.csv')!=bm['per_image_sha256']:
            raise ValueError('Boundary output hash mismatch')
        name=meta['model']['model'];asset=meta['model']['asset']
        label='direct_metric' if asset['alignment']=='none' else 'relative_GT_aligned'
        table.append({'model':name,'evaluation':label,'images':scores['images'],
            'contributing_images':scores['contributing_images']['rmse_m'],'alignment_failures':scores.get('alignment_failures',0),
            'coverage':scores['pixel_pooled_coverage'],**scores['image_macro_metrics_on_valid'],
            **{key+'_full_gt':v for key,v in scores['image_macro_full_gt_delta'].items()},
            'boundary_precision':boundary['pooled_precision'],'boundary_recall':boundary['pooled_recall'],
            'boundary_f1':boundary['pooled_f1'],'boundary_capped_distance_px':boundary['gt_edge_weighted_capped_distance_px'],
            'mean_seconds':scores['mean_infer_seconds'],'fps_excluding_io':scores['throughput_fps_excluding_io'],
            'peak_allocated_gib':meta['peak_cuda_allocated_bytes']/2**30,'peak_reserved_gib':meta['peak_cuda_reserved_bytes']/2**30,
            'parameters':meta['model']['parameters'],'checkpoint_bytes':meta['model']['weight_bytes']})
        runs.append((run,meta))
        provenance.append({'run':source,'boundary_run':bsource,'metadata_sha256':checksum(run/'metadata.json'),
            'summary_sha256':checksum(run/'summary.json'),'per_image_sha256':checksum(run/'per_image.csv'),
            'model':meta['model'],'identity':meta['identity'],'code_commit':meta['code_commit'],
            'boundary_metadata':bm,'summary':scores,'boundary_summary':boundary})
    if len({tuple(m['identity']['sample_ids']) for _,m in runs})!=1:
        raise ValueError('Models use different test IDs')
    output.mkdir(parents=True)
    write_csv(output/'summary.csv',table)
    for (run,meta),bsource in zip(runs,a.boundary):
        name=meta['model']['model']
        shutil.copyfile(run/'per_image.csv',output/f'{name}_per_image.csv')
        shutil.copyfile(repo_path(bsource)/'per_image.csv',output/f'{name}_boundaries.csv')
    uni=next(run for run,meta in runs if meta['model']['model']=='unidepth')
    with (uni/'per_image.csv').open(encoding='utf-8',newline='') as f:
        ordered=sorted(csv.DictReader(f),key=lambda r:float(r['rmse_m']))
    selections=[(ordered[len(ordered)//2]['sample_id'],'median UniDepth RMSE'),(ordered[-1]['sample_id'],'largest UniDepth RMSE')]
    failures=[]
    for run,meta in runs:
        if meta['model']['asset']['alignment']=='none':
            continue
        for sid in meta['identity']['sample_ids']:
            record=json.loads((run/'records'/f'{sid}.json').read_text(encoding='utf-8'))
            if record['transformation'].get('alignment_status')=='failed':
                failures.append(sid)
    if failures:
        selections.append((sorted(failures)[0],'relative alignment failure'))
    data=repo_path(read_yaml(repo_path(a.paths_config))['data_roots']['nyuv2'])
    cmap=plt.colormaps['viridis'].copy();cmap.set_bad('#b4b4b4')
    fig,axes=plt.subplots(len(selections),2+len(runs),figsize=(15,3.1*len(selections)),squeeze=False,layout='constrained')
    for row,(sid,reason) in enumerate(selections):
        record=json.loads((uni/'records'/f'{sid}.json').read_text(encoding='utf-8'))
        with Image.open(data/record['rgb_path']) as img:
            axes[row,0].imshow(img.convert('RGB'))
        axes[row,0].set_title(f'{sid} | {reason}',fontsize=8)
        gt=np.load(data/record['gt_path'],allow_pickle=False)
        axes[row,1].imshow(gt,cmap=cmap,vmin=0,vmax=10);axes[row,1].set_title('GT Z depth',fontsize=9)
        for col,(run,meta) in enumerate(runs,2):
            raw=np.load(run/'predictions'/f'{sid}.npy',allow_pickle=False)
            valid=np.load(run/'masks'/f'{sid}.npy',allow_pickle=False)
            asset=meta['model']['asset']
            try:
                depth,mask,_=prepare_prediction(raw,gt,evaluation_mask(gt),asset['prediction_type'],asset['unit'],asset['alignment'],valid)
            except RelativeAlignmentError:
                depth=np.full_like(gt,np.nan)
                axes[row,col].text(.5,.5,'Alignment failed',ha='center',va='center',transform=axes[row,col].transAxes,fontsize=9)
            im=axes[row,col].imshow(depth,cmap=cmap,vmin=0,vmax=10)
            suffix=' (GT aligned)' if asset['alignment']!='none' else ' (direct metric)'
            axes[row,col].set_title(meta['model']['model']+suffix,fontsize=9)
        for ax in axes[row]:
            ax.axis('off')
    fig.colorbar(im,ax=axes[:,1:].ravel().tolist(),shrink=.65,label='Z depth (m), fixed display 0-10; raw values unchanged')
    fig.suptitle('NYUv2 standard test: deterministic case selection rules; gray = unavailable',fontsize=12)
    fig.savefig(output/'comparison.png',dpi=180);fig.savefig(output/'comparison.pdf');plt.close(fig)
    write_json(output/'provenance.json',{'runs':provenance,'case_selection':selections,'alignment_failure_ids':failures,
        'visualization':'full RGB/GT/prediction dimensions; metrics separately apply fixed evaluation crop; shared 0-10 m display',
        'summary_sha256':checksum(output/'summary.csv'),'figure_sha256':checksum(output/'comparison.png'),
        'exporter_sha256':checksum(ROOT/'scripts/summarize_nyuv2.py')})
    print(f'PASS: {len(runs)} complete standard tests; selected cases {selections}; failures={failures}')


if __name__=='__main__':
    main()
