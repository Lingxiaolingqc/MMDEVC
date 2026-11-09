"""Render report figures from independently measured results, never invented values.

Run from independent/: .venv/Scripts/python scripts/make_figures.py
"""
from pathlib import Path
import json

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
import numpy as np

from depth_project.cache import load_predictions
from depth_project.data import load_nyu_sample,read_pfm
from depth_project.render import colorize
from depth_project.video import iter_rgb_frames
from depth_project.flow import estimate_pair
from depth_project.fusion import update
from depth_project.records import Frame,DepthPrediction
from depth_project.warp import warp_previous


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/figures'
OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':8,'axes.titlesize':9,'xtick.labelsize':7,'ytick.labelsize':7,'savefig.dpi':220})
COLORS=['#8b9098','#cf8a36','#459ca8','#5473b8','#8555a1']


def save(fig,name):
    fig.savefig(OUT/name,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def pipeline():
    fig,ax=plt.subplots(figsize=(7.2,2.7));ax.set_xlim(0,10);ax.set_ylim(0,4);ax.axis('off')
    boxes={
        'rgb':(0.1,2.5,1.5,0.8,'RGB frames\n$t-1$, $t$','#eef0f4'),
        'depth':(2.15,2.5,2.0,0.8,'Frozen DA V2-S\ncurrent $q_t$','#e9edf6'),
        'flow':(2.15,0.5,2.0,0.8,'Forward + backward\nFarneback flow','#e6f2f1'),
        'warp':(4.7,0.5,1.8,0.8,'Warp history\nreject occlusions','#e6f2f1'),
        'gate':(4.7,2.5,1.8,0.8,'Photo + depth\nconfidence gates','#f1eaf6'),
        'fusion':(7.05,1.5,1.5,0.8,'Weighted fusion\n$(1-w)q_t+w h_t$','#eee3f3'),
        'out':(9.0,1.5,0.85,0.8,'Relative\ndepth +\nblur','#f0eef5'),
    }
    for x,y,w,h,text,color in boxes.values():
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',fc=color,ec='#707883',lw=0.8))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8)
    def arrow(start,end):ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=11,color='#535c69',lw=1,connectionstyle='arc3,rad=0'))
    arrow((1.7,2.9),(2.05,2.9));arrow((1.0,2.4),(2.05,0.9))
    arrow((4.25,2.9),(4.6,2.9));arrow((4.25,0.9),(4.6,0.9))
    arrow((6.6,0.9),(7.1,1.5));arrow((6.6,2.9),(7.1,2.25));arrow((8.65,1.9),(8.9,1.9))
    ax.text(5.55,0.03,'Previous fused depth is the causal history; newly revealed pixels use current depth.',ha='center',fontsize=7,color='#4e5663')
    ax.text(0.1,3.75,'Independent implementation: the pretrained depth network stays frozen',fontsize=10,weight='bold')
    save(fig,'pipeline.png')


def ablations():
    final=json.loads((ROOT/'results/final-evaluation.json').read_text())
    rows=final['rows'];names=['Frame-wise','EMA','Flow','Occlusion','Adaptive']
    base=rows[0]['temporal']
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.5),gridspec_kw={'width_ratios':[1.2,1.2,1]})
    axes[0].bar(range(5),[100*r['temporal']/base for r in rows],color=COLORS)
    axes[0].axhline(100,color='#444',lw=0.6,ls='--');axes[0].set_ylabel('Temporal proxy (% of raw)');axes[0].set_title('Lower is smoother')
    axes[1].bar(range(5),[100*r['edge_retention'] for r in rows],color=COLORS)
    axes[1].set_ylabel('Baseline-edge strength (%)');axes[1].set_title('A diagnostic, not GT accuracy')
    axes[2].bar(range(5),[r['synthetic_settling_frames'] for r in rows],color=COLORS)
    axes[2].set_ylabel('Synthetic step settling (frames)');axes[2].set_title('Lower is more responsive')
    for ax in axes:
        ax.set_xticks(range(5),names,rotation=45,ha='right');ax.grid(axis='y',alpha=0.15);ax.set_axisbelow(True)
    fig.tight_layout(w_pad=1.3)
    save(fig,'ablations.png')


def qualitative():
    final=json.loads((ROOT/'runs/final/summary.json').read_text())
    baseline,_=load_predictions(ROOT/'runs/baselines/rollercoaster-dav2.npz')
    reference,_=load_predictions(ROOT/'runs/baselines/rollercoaster-vda-392.npz')
    selected,_=load_predictions(ROOT/'runs/final/adaptive.npz')
    frames=list(iter_rgb_frames(ROOT/'third_party/Video-Depth-Anything/assets/example_videos/davis_rollercoaster.mp4'))
    indices=[0,35,69]
    ranges=final['display_ranges']
    fig,axes=plt.subplots(3,4,figsize=(7.2,3.45))
    titles=['RGB','DA V2-S (relative)','VDA-S (relative)','Adaptive fusion']
    for row,index in enumerate(indices):
        rgb=cv2.resize(frames[index].rgb,(640,360),interpolation=cv2.INTER_AREA)
        images=[rgb,colorize(baseline[index],ranges['DA V2 Small (relative)']),colorize(reference[index],ranges['VDA Small (relative)']),colorize(selected[index],ranges['Flow-guided fusion (relative)'])]
        for col,image in enumerate(images):
            axes[row,col].imshow(image);axes[row,col].axis('off')
            if row==0:axes[row,col].set_title(titles[col],fontsize=8)
        axes[row,0].text(-0.03,0.5,f'Frame {index}',transform=axes[row,0].transAxes,ha='right',va='center',rotation=90,fontsize=7)
    fig.subplots_adjust(wspace=0.015,hspace=0.02)
    save(fig,'video-comparison.png')
    # One fixed noninitial frame/crop shows thin supports and a moving boundary.
    crop=(slice(45,190),slice(80,470))
    fig,axes=plt.subplots(1,4,figsize=(7.2,1.15))
    index=35
    rgb=cv2.resize(frames[index].rgb,(640,360),interpolation=cv2.INTER_AREA)
    occlusion=load_predictions(ROOT/'runs/final/occlusion.npz')[0]
    chosen=[('RGB',rgb),('DA V2-S',colorize(baseline[index],ranges['DA V2 Small (relative)'])),('Occlusion-only',colorize(occlusion[index],ranges['DA V2 Small (relative)'])),('Adaptive',colorize(selected[index],ranges['DA V2 Small (relative)']))]
    for ax,(title,image) in zip(axes,chosen):
        ax.imshow(image[crop]);ax.axis('off');ax.set_title(title,fontsize=8)
    fig.tight_layout(pad=0.2)
    save(fig,'boundary-zooms.png')


def nyu_failures():
    summary=json.loads((ROOT/'runs/nyu/summary.json').read_text())
    worst=sorted(summary['per_image'],key=lambda r:r['abs_rel'],reverse=True)[:3]
    fig,axes=plt.subplots(3,4,figsize=(7.2,4.5))
    records=[]
    for row,record in enumerate(worst):
        index=record['image_index_zero_based']
        rgb,gt=load_nyu_sample(index,ROOT/'data/nyu/nyu_depth_v2_labeled.mat')
        with np.load(ROOT/f'runs/nyu/prediction_{index:04d}.npz') as source:pred=source['depth_m']
        valid=np.isfinite(gt)&(gt>0.001)&(gt<10)
        evaluated=np.zeros(valid.shape,bool);evaluated[45:471,41:601]=True;valid &= evaluated
        error=np.full(gt.shape,np.nan,np.float32);error[valid]=np.abs(np.clip(pred[valid],0.001,10)-gt[valid])/gt[valid]
        axes[row,0].imshow(rgb)
        axes[row,1].imshow(gt,cmap='inferno',vmin=0,vmax=10)
        axes[row,2].imshow(pred,cmap='inferno',vmin=0,vmax=10)
        axes[row,3].imshow(np.ma.masked_invalid(error),cmap='magma',vmin=0,vmax=0.5)
        for ax in axes[row]:ax.axis('off')
        axes[row,0].set_title(f'NYU index {index}',fontsize=8)
        axes[row,3].set_title(f'AbsRel {record["abs_rel"]:.3f}',fontsize=8)
        records.append(record)
    for ax,title in zip(axes[0],['RGB','Filled GT, 0-10 m','UniDepthV2-S, 0-10 m','Pixel AbsRel, 0-0.5']):ax.text(0.5,1.22,title,transform=ax.transAxes,ha='center',fontsize=8)
    fig.subplots_adjust(top=0.88,bottom=0.02,left=0.02,right=0.99,hspace=0.35,wspace=0.035)
    save(fig,'nyu-failures.png')
    (ROOT/'results/failure-selection.json').write_text(json.dumps({'selection':'three highest test per-image AbsRel values; no manual exclusion','images':records},indent=2))


def stereo_figure():
    fig,axes=plt.subplots(2,4,figsize=(7.2,3.9))
    for row,name in enumerate(['Adirondack','Jadeplant']):
        pair=ROOT/f'data/middlebury/MiddEval3/trainingQ/{name}'
        rgb=cv2.cvtColor(cv2.imread(str(pair/'im0.png')),cv2.COLOR_BGR2RGB)
        gt=read_pfm(pair/'disp0GT.pfm')
        with np.load(ROOT/f'runs/stereo/{name}.npz') as source:pred=source['disparity_px']
        valid=np.isfinite(gt)&(gt>0)&np.isfinite(pred)&(pred>0)
        limit=np.percentile(gt[np.isfinite(gt)&(gt>0)],98)
        axes[row,0].imshow(rgb)
        axes[row,1].imshow(np.ma.masked_invalid(gt),cmap='inferno',vmin=0,vmax=limit)
        axes[row,2].imshow(np.ma.masked_where(pred<=0,pred),cmap='inferno',vmin=0,vmax=limit)
        axes[row,3].imshow(np.ma.masked_where(~valid,np.abs(gt-pred)),cmap='magma',vmin=0,vmax=5)
        for ax in axes[row]:ax.axis('off')
        axes[row,0].set_title(name,fontsize=8)
    for ax,title in zip(axes[0],['Left RGB','GT disparity','StereoSGBM','Absolute error, 0-5 px']):ax.text(0.5,1.2,title,transform=ax.transAxes,ha='center',fontsize=8)
    fig.tight_layout(pad=0.25)
    save(fig,'stereo-comparison.png')


def flow_debug():
    """Fixed development frame 35; reconstruct causal state with the frozen config."""
    config=json.loads((ROOT/'configs/final.json').read_text())
    values,_=load_predictions(ROOT/'runs/baselines/tokyo-dav2.npz')
    source=ROOT/'third_party/Video-Depth-Anything/assets/example_videos/Tokyo-Walk_rgb.mp4'
    frames=list(iter_rgb_frames(source,36))
    state=None
    for index,frame in enumerate(frames):
        rgb=cv2.resize(frame.rgb,(values.shape[2],values.shape[1]),interpolation=cv2.INTER_AREA)
        prediction=DepthPrediction(values[index],'relative','dav2','diagnostic','official','fp16')
        flows=None if state is None else estimate_pair(state.rgb,rgb,config['flow'])
        previous=state
        _,state,diagnostics=update(Frame(index,frame.timestamp_s,rgb),prediction,state,config,flows=flows)
    _,backward=flows
    warped,_=warp_previous(previous.depth,backward)
    limits=np.percentile(values,[2,98])
    fig,axes=plt.subplots(2,3,figsize=(7.2,3.4))
    axes[0,0].imshow(previous.rgb);axes[0,0].set_title('Previous RGB, frame 34')
    axes[0,1].imshow(rgb);axes[0,1].set_title('Current RGB, frame 35')
    axes[0,2].imshow(rgb)
    yy,xx=np.mgrid[12:rgb.shape[0]:28,12:rgb.shape[1]:28]
    axes[0,2].quiver(xx,yy,backward[yy,xx,0],backward[yy,xx,1],angles='xy',scale_units='xy',scale=1,color='#ffe868',width=.004)
    axes[0,2].set_title('Backward flow, pixel displacements')
    axes[1,0].imshow(colorize(warped,limits));axes[1,0].set_title('Warped recurrent history')
    axes[1,1].imshow(diagnostics['valid'],cmap='gray',vmin=0,vmax=1);axes[1,1].set_title('Accepted support (white)')
    axes[1,2].imshow(diagnostics['history_weight'],cmap='viridis',vmin=0,vmax=.75)
    axes[1,2].set_title('Effective history weight, 0-0.75')
    for ax in axes.flat:ax.axis('off')
    fig.tight_layout(pad=.3,h_pad=1)
    save(fig,'flow-diagnostics.png')


if __name__=='__main__':
    pipeline();ablations();qualitative();nyu_failures();stereo_figure();flow_debug()
    print('Saved report figures:',*[p.name for p in OUT.glob('*.png')],sep='\n')
