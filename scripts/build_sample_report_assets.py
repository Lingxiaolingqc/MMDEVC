"""Build compact, source-verified figures and numeric evidence for the sample report."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import argparse,csv,json,shutil,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image
from _common import ROOT
from _video_review import read_json,read_rows,contact_sheet
from p3_depth.application import background_blur
from p3_depth.data.nyuv2 import evaluation_mask

OUT=ROOT/'report/sample_20261109'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def arr(p): return np.load(p,mmap_mode='r',allow_pickle=False)
SOURCES={}
def track(p):
    SOURCES[p.relative_to(ROOT).as_posix()]=sha(p)
    return p
def json_at(p): return read_json(track(p))
def copy(p,dest):
    shutil.copyfile(track(p),dest)
def grid(images,titles,path,figsize):
    fig,axes=plt.subplots(len(images),len(images[0]),figsize=figsize,layout='constrained',squeeze=False)
    for r,line in enumerate(images):
        for c,im in enumerate(line):
            axes[r,c].imshow(im); axes[r,c].set_title(titles[r][c],fontsize=10);axes[r,c].axis('off')
    fig.savefig(path,dpi=200);plt.close(fig)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-draft',action='store_true',help='Rebuild only this sample report draft; source experiments remain unchanged')
    args=parser.parse_args()
    if (OUT/'asset_provenance.json').exists() and not args.refresh_draft: raise FileExistsError('Preserve verified asset build; explicitly --refresh-draft for report edits.')
    (OUT/'images').mkdir(exist_ok=True);(OUT/'data').mkdir(exist_ok=True)
    analysis={
    'stereo':'outputs/part1_baseline/stereo/all/analysis',
    'nyu':'outputs/part2_models/unidepth/test/analysis',
    'da':'outputs/part1_baseline/dav2_relative/test/course_rollercoaster_70/analysis',
    'vda':'outputs/part2_models/vda_relative/test/course_rollercoaster_70/analysis',
    'main_dev':'outputs/part3_improvement/dev_candidates/alpha070/dev_tokyo_100/analysis',
    'main_course':'outputs/part3_improvement/test/course_rollercoaster_70/analysis',
    'history_dev':'outputs/part3_improvement/supplementary/dev/dev_tokyo_100/history/analysis',
    'history_course':'outputs/part3_improvement/supplementary/test/course_rollercoaster_70/history/analysis',
    'confidence_dev':'outputs/part3_improvement/supplementary/dev/dev_tokyo_100/confidence/analysis',
    'confidence_course':'outputs/part3_improvement/supplementary/test/course_rollercoaster_70/confidence/analysis',
    'candidates':'outputs/part3_improvement/selection/analysis',
    'application':'outputs/application_exploratory/course_rollercoaster_70/transition010_030/analysis'}
    for name,folder in analysis.items():
        root=ROOT/folder; review=json_at(root/'review.json')
        q=json_at(root/'quantitative_review.json')
        if sha(root/'quantitative_review.json') != review.get('quantitative_source_sha256',review.get('quantitative_review_sha256')):
            raise ValueError('Changed quantitative review '+name)
        if sha(root/'analysis.md') != review.get('analysis_markdown_sha256',review.get('analysis_sha256')):
            raise ValueError('Changed analysis '+name)
        for file,digest in {**q['artifacts_sha256'],**review.get('artifacts_sha256',{})}.items():
            if sha(root/file)!=digest: raise ValueError('Changed reviewed artifact '+str(root/file))
        # Narratives keep their figures and relative links in the original analysis
        # directory. The portable report includes compact tables/review records.
        track(root/'analysis.md')
        copy(root/'review.json',OUT/'data'/f'{name}_review.json')
        for p in sorted(root.glob('*.csv')): copy(p,OUT/'data'/f'{name}_{p.name}')
        if (root/'summary.json').exists():copy(root/'summary.json',OUT/'data'/f'{name}_summary.json')
    # Keep original official template and bibliography unmodified.
    for name in ('cvpr.sty','ieeenat_fullname.bst','references.bib'): copy(ROOT/'report/paper'/name,OUT/name)
    # Compact distribution boxplots: quantiles are from all valid pixels, no sampling.
    stereo=read_rows(track(ROOT/analysis['stereo']/'error_distributions.csv'))
    selected=['Adirondack','Jadeplant','PianoL','Pipes','Teddy']
    fig,axes=plt.subplots(2,1,figsize=(5.5,3.9),layout='constrained')
    for ax,prefix,unit in zip(axes,('disparity','depth'),('px','m')):
        items=[];labels=[]
        for name in selected:
            r=next(r for r in stereo if r['sample_id']==name and r['custom_lr']=='on')
            items.append(dict(med=float(r[prefix+'_median']),q1=float(r[prefix+'_q25']),q3=float(r[prefix+'_q75']),whislo=float(r[prefix+'_p05']),whishi=float(r[prefix+'_p95']),fliers=[]))
            labels.append(name+'\n'+f"{100*float(r['coverage']):.1f}%")
        ax.bxp(items,showfliers=False);ax.set_xticks(range(1,6),labels,fontsize=11)
        ax.set_ylabel('Absolute error ('+unit+')',fontsize=12);ax.tick_params(axis='y',labelsize=12);ax.grid(axis='y',alpha=.2)
    fig.savefig(OUT/'images/stereo_boxes.png',dpi=200);plt.close(fig)
    # A selected existing full eight-panel diagnostic, unchanged.
    jade=ROOT/'outputs/part1_baseline/stereo/all/Jadeplant/visualizations'
    pngs=list(jade.glob('*.png'))
    preferred=next((p for p in pngs if p.name=='visualization.png'),None)
    if preferred is None:
        if len(pngs)!=1: raise ValueError('Ambiguous Jadeplant visual '+str(pngs))
        preferred=pngs[0]
    copy(preferred,OUT/'images/jadeplant.png')
    # NYU: shared meter and error scales, explicit crop.
    rows={r['sample_id']:r for r in read_rows(track(ROOT/'manifests/nyuv2.csv'))}
    fig,axes=plt.subplots(2,4,figsize=(11,4.5),layout='constrained')
    for i,sid in enumerate(('nyuv2_0088','nyuv2_0033')):
        r=rows[sid];rgb=np.array(Image.open(track(ROOT/'data/nyuv2'/r['rgb_path'])))
        gt=arr(track(ROOT/'data/nyuv2'/r['depth_gt_path']))
        pred=np.clip(arr(track(ROOT/'outputs/part2_models/unidepth/test/predictions'/(sid+'.npy'))),.001,10)
        mask=evaluation_mask(gt);err=np.ma.masked_where(~mask,np.abs(pred-gt))
        axes[i,0].imshow(rgb)
        high,errhigh=(10,2) if i==0 else (2.5,.5)
        axes[i,1].imshow(gt,vmin=0,vmax=high,cmap='viridis')
        axes[i,2].imshow(pred,vmin=0,vmax=high,cmap='viridis')
        im=axes[i,3].imshow(err,vmin=0,vmax=errhigh,cmap='magma')
        for ax,title in zip(axes[i],(sid+' RGB',f'GT (0-{high:g} m)',f'Prediction (0-{high:g} m)',f'|error| (0-{errhigh:g} m)')):
            ax.set_title(title,fontsize=10);ax.axis('off')
    fig.savefig(OUT/'images/nyu_cases.png',dpi=200);plt.close(fig)
    data=read_rows(track(ROOT/analysis['nyu']/'per_image.csv'))
    fig,axes=plt.subplots(1,3,figsize=(4.8,2.2),layout='constrained')
    for ax,k,title in zip(axes,('abs_rel','rmse_m','delta1'),('AbsRel','RMSE (m)','delta1')):
        q=np.quantile([float(r[k]) for r in data],[.05,.25,.5,.75,.95])
        ax.bxp([dict(whislo=q[0],q1=q[1],med=q[2],q3=q[3],whishi=q[4],fliers=[])],showfliers=False)
        ax.set_title(title);ax.set_xticks([]);ax.grid(axis='y',alpha=.2)
    fig.savefig(OUT/'images/nyu_distribution.png',dpi=200);plt.close(fig)
    # Video panels from unchanged float arrays, never per-frame normalization.
    clip='course_rollercoaster_70'; cache=ROOT/'outputs/shared/video_cache'/clip
    frames=arr(track(cache/'frames.npy')); times=[float(r['timestamp_s']) for r in read_rows(track(cache/'frames.csv'))]
    runs=[ROOT/'outputs/part1_baseline/dav2_relative/test'/clip, ROOT/'outputs/part3_improvement/test'/clip/'E4', ROOT/'outputs/part2_models/vda_relative/test'/clip]
    d=[arr(track(p/'depths.npy')) for p in runs];v=[arr(track(p/'valid.npy')) for p in runs]
    limits=[json_at(ROOT/'outputs/part3_improvement/test'/clip/f'E{x}/metrics.json')['fixed_clip_percentile_limits'] for x in (0,4,5)]
    contact_sheet(frames,d,v,limits,['DA','E4','VDA'],times,[14,15,16],OUT/'images/video.png',roi=[100,60,450,260])
    base=ROOT/'outputs/part3_improvement/test'/clip
    methods=['E1','E2','E3','E4']
    contact_sheet(frames,[arr(track(base/m/'depths.npy')) for m in methods],[arr(track(base/m/'valid.npy')) for m in methods],
                  [limits[0]]*4,methods,times,[15,35],OUT/'images/modules.png',roi=[100,60,450,260])
    hist=ROOT/'outputs/part3_improvement/supplementary/test'/clip/'history/E1_pairwise'
    contact_sheet(frames,[arr(track(base/'E1/depths.npy')),arr(track(hist/'depths.npy'))],
                  [arr(track(base/'E1/valid.npy')),arr(track(hist/'valid.npy'))],[limits[0]]*2,
                  ['E1 recursive','E1 pairwise'],times,[15],OUT/'images/history.png',roi=[100,60,450,260])
    # Application recomputed by exactly the original function, same params.
    params=[(.45,.70),(.10,.30)];ims=[];titles=[]
    for t in (0,35):
        line=[frames[t]]
        for prediction,mask,threshold in ((d[0],v[0],params[0]),(d[0],v[0],params[1]),(d[1],v[1],params[1])):
            line.append(background_blur(frames[t],prediction[t],mask[t],limits[0],sigma=5,transition=threshold)[0])
        ims.append([im[50:270,70:500] for im in line])
        titles.append([f'RGB f{t}','DA old threshold','DA new threshold','E4 new threshold'])
    grid(ims,titles,OUT/'images/application.png',(11,3.4))
    provenance={'date':'2026-11-09','scope':'Report figures from reviewed current artifacts; no inference or parameter changes.',
        'sources_sha256':SOURCES,'implementation_sha256':sha(Path(__file__)),
        'figure_selection':{'stereo_boxes':selected,'nyu_cases':['nyuv2_0088','nyuv2_0033'],
        'video':{'frames':[14,15,16],'roi':[100,60,450,260]},'modules':{'frames':[15,35],'roi':[100,60,450,260]},
        'history':{'frames':[15],'roi':[100,60,450,260]},'application':{'frames':[0,35],'roi':[70,50,500,270]}},
        'images_sha256':{p.name:sha(p) for p in sorted((OUT/'images').glob('*.png'))}}
    (OUT/'asset_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    print('PASS sample assets:',len(provenance['images_sha256']),'figures;',len(SOURCES),'source hashes')
if __name__=='__main__':main()
