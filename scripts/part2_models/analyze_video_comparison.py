"""Compare saved DA/VDA outputs on identical frames; no inference or GT alignment."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from _common import ROOT,repo_path,checksum,write_csv,write_json
from _video_review import read_json,read_rows,verify_preview,contact_sheet,selected_windows
from part3_improvement.run_supplementary_ablations import load_sources
from part3_improvement.run_fusion_suite import load_run
from p3_depth.evaluation.temporal import evaluate_relative_video


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',default='outputs/shared/video_cache/course_rollercoaster_70')
    p.add_argument('--baseline',default='outputs/part1_baseline/dav2_relative/test/course_rollercoaster_70')
    p.add_argument('--vda',default='outputs/part2_models/vda_relative/test/course_rollercoaster_70')
    p.add_argument('--flow',default='outputs/part3_improvement/flow/test/course_rollercoaster_70')
    p.add_argument('--reference-suite',default='outputs/part3_improvement/test/course_rollercoaster_70')
    p.add_argument('--output-dir')
    p.add_argument('--roi',nargs=4,type=int,default=[100,60,450,260])
    a=p.parse_args()
    cache,baseline,vda,flow,suite=map(repo_path,(a.cache,a.baseline,a.vda,a.flow,a.reference_suite))
    out=repo_path(a.output_dir) if a.output_dir else vda/'analysis'
    if out.exists(): raise FileExistsError('Use a new directory; preserve prior analysis')
    source=load_sources(cache,baseline,flow,suite)
    vm,vpred,vvalid=load_run(vda,source['cm'])
    if vm['model']['model']!='vda_relative' or vm['prediction_type']!='inverse_depth_like' or checksum(vda/'metadata.json')!=source['sm']['vda_metadata_sha256']:
        raise ValueError('Require the same relative VDA source as the verified suite')
    arrays=[source['base'],vpred]; masks=[source['valid'],vvalid]
    limits=[source['limits'],list(map(float,np.quantile(vpred[vvalid],[.01,.99])))]
    all_rows=[]; resources=[]; pair_rows=[]; stats=[]; previews={}
    for name,folder,pred,valid,span,meta,variant in zip(('DA','VDA'),(baseline,vda),arrays,masks,limits,(source['bm'],vm),('E0','E5')):
        scores,pairs=evaluate_relative_video(pred,valid,source['fields']['backward'],source['fields']['common_mask'],source['times'],normalization_limits=span)
        if scores!=read_json(suite/variant/'metrics.json'): raise ValueError('Saved video diagnostics differ: '+name)
        saved=read_rows(suite/variant/'per_pair.csv')
        if len(saved)!=len(pairs): raise ValueError('Missing pair scores')
        for row,old in zip(pairs,saved):
            for k,value in row.items():
                if value is not None and not np.isclose(value,float(old[k]),rtol=1e-10,atol=1e-12): raise ValueError('Changed pair statistic: '+k)
        previews[name]=verify_preview(cache,folder,source['times'],span)
        stats.append(scores); pair_rows.append(pairs)
        all_rows.extend({'model':name,**r} for r in pairs)
        resources.append({'model':name,'frames':len(pred),'total_seconds':meta['inference_seconds'],
            'ms_per_frame':meta['mean_seconds_per_frame']*1000,'fps':meta['fps'],
            'peak_cuda_allocated_mib':meta['peak_cuda_allocated_bytes']/1024**2,
            'peak_cuda_reserved_mib':meta['peak_cuda_reserved_bytes']/1024**2,
            'input_nchw':str(meta['model']['encoder_shapes_nchw'][0]),
            'precision':'FP16','device':meta['model']['device_name'],
            'T_own_fixed_span':scores['capped_normalized_l1_full_reference'],
            'raw_relative_l1':scores['temporal_l1_relative_on_valid'],
            'normalization_low':span[0],'normalization_high':span[1],
            'reference_pixels':scores['reference_pixels'],'pair_coverage':scores['pair_coverage']})
    out.mkdir(parents=True); (out/'figures').mkdir()
    write_csv(out/'comparison.csv',resources); write_csv(out/'per_pair.csv',all_rows)
    write_json(out/'model_resources.json',{'DA':source['bm'],'VDA':vm})
    fig,axes=plt.subplots(1,2,figsize=(12,4.2),layout='constrained')
    for name,scores,pairs in zip(('DA','VDA'),stats,pair_rows):
        axes[0].plot([r['timestamp_s'] for r in pairs],[r['capped_normalized_l1_full_reference'] for r in pairs],label=name)
        axes[1].plot(source['times'],np.array(scores['frame_median_relative'])/scores['normalization_span'],label=name)
    axes[0].set(title='Flow-aligned change T, own fixed spans',xlabel='Time (s)',ylabel='Normalized relative variation')
    axes[1].set(title='Whole-frame median / own span',xlabel='Time (s)',ylabel='Normalized median (background dominated)')
    for ax in axes: ax.legend(); ax.grid(alpha=.2)
    fig.suptitle('Same RGB-derived reference mask | diagnostics, not depth accuracy\n'
                 'Different model scales and scene/motion changes limit cross-model interpretation')
    fig.savefig(out/'figures/diagnostics.png',dpi=180); plt.close(fig)
    frames=np.load(cache/'frames.npy',mmap_mode='r',allow_pickle=False)
    overview=np.linspace(0,len(frames)-1,6).round().astype(int).tolist()
    windows=selected_windows(pair_rows[0],len(frames))
    # Original VDA code blends eight output frames at these first two join regions.
    for start in (24,46):
        if start+8<=len(frames): windows.append(list(range(start,start+8)))
    windows=sorted(windows,key=lambda w:w[0])
    args=(frames,arrays,masks,limits,['DA','VDA'],source['times'])
    contact_sheet(*args,overview,out/'figures/overview.png')
    for w in windows: contact_sheet(*args,w,out/f'figures/roi_{w[0]}_{w[-1]}.png',roi=a.roi)
    write_json(out/'quantitative_review.json',{'status':'quantitative_complete_visual_pending','frames':len(frames),
        'source_metadata_sha256':{p.relative_to(ROOT).as_posix():checksum(p) for p in
          (cache/'metadata.json',baseline/'metadata.json',vda/'metadata.json',flow/'metadata.json',suite/'metadata.json',suite/'verification.json')},
        'implementation_sha256':{p:checksum(ROOT/p) for p in ('scripts/part2_models/analyze_video_comparison.py','scripts/_video_review.py','src/p3_depth/evaluation/temporal.py','third_party/Video-Depth-Anything/video_depth_anything/video_depth.py')},
        'preview_checks':previews,'model_summaries':dict(zip(('DA','VDA'),stats)),
        'overview_frames':overview,'roi_xyxy':a.roi,'consecutive_windows':windows,
        'selection':'Same DA diagnostic windows plus the first two official VDA interpolation regions; post-hoc, not random. Fixed image ROI is not object tracking.',
        'limitations':'No depth GT or latency labels; each model has its own fixed whole-clip display scale. Raw units differ. VDA is offline, not a causal streaming method. Timing is the recorded inference scope, not application FPS.',
        'raw_sources_modified':False,
        'artifacts_sha256':{p.relative_to(out).as_posix():checksum(p) for p in out.rglob('*') if p.is_file()}})
    print({'frames':len(frames),'windows':windows,'output':str(out)},flush=True)


if __name__=='__main__': main()
