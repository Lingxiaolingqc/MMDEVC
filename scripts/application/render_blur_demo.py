"""Export RGB, relative depth and matched baseline/improved background-blur effects."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
from fractions import Fraction
import json
import av
import numpy as np
from PIL import Image

from _common import ROOT, repo_path, checksum, write_json, write_csv
from part3_improvement.run_fusion_suite import load_run
from render_depth_video import colorize,text
from p3_depth.application import background_blur
from _paths import clip_output


def checked_reference(folder, cache_hash, method_hashes, limits):
    """Bind a settings comparison to a preserved application on the same predictions."""
    meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    if (meta.get('status')!='complete' or meta.get('application')!='relative_background_blur'
            or meta.get('cache_metadata_sha256')!=cache_hash
            or {m['name']:m['metadata_sha256'] for m in meta['methods']}!=method_hashes
            or meta['parameters']['limits']!=limits):
        raise ValueError('Reference application must use the same cache, predictions and fixed scale')
    # The effect implementation must be identical; only its parameters may change.
    if meta['implementation_sha256']['src/p3_depth/application.py']!=checksum(ROOT/'src/p3_depth/application.py'):
        raise ValueError('Reference effect implementation changed; cannot isolate settings')
    for name,key in [('demo.mp4','demo.mp4'),('preview.png','preview.png'),('blur_strength.csv','blur_strength.csv')]:
        if checksum(folder/name)!=meta['files'][key]:
            raise ValueError('Reference application artifact changed')
    return meta


def settings_comparison(frames,timestamps,base,bvalid,improved,ivalid,limits,previous,current,output,fps):
    """Two rows of old/new settings and two columns of unchanged DA/frozen E4."""
    h,w=frames.shape[1:3];row_height=h+72;previews=[];rows=[]
    time_base=Fraction(1,1000000);selected={0,len(frames)//2,len(frames)-1}
    with av.open(str(output/'settings_comparison.mp4'),'w') as container:
        stream=container.add_stream('libx264',rate=Fraction(str(fps)))
        stream.width,stream.height,stream.pix_fmt=2*w,2*row_height,'yuv420p'
        stream.time_base=time_base;stream.codec_context.time_base=time_base
        stream.options={'crf':'19','preset':'fast'}
        for i,rgb in enumerate(frames):
            canvas=np.full((2*row_height,2*w,3),24,np.uint8);effects=[];weights=[]
            for row,params in enumerate((previous,current)):
                for column,(depth,valid,name) in enumerate(((base,bvalid,'DA'),(improved,ivalid,'E4'))):
                    effect,weight=background_blur(rgb,depth[i],valid[i],limits,
                        sigma=params['sigma_px'],transition=params['transition_normalized'])
                    effects.append(effect);weights.append(weight)
                    x,y=column*w,row*row_height
                    canvas[y+30:y+30+h,x:x+w]=effect
                    label='Previous settings' if row==0 else 'Adjusted settings'
                    text(canvas,f'{name} | {label}',x+8,y+21)
                    start,end=params['transition_normalized']
                    text(canvas,f'frame {i} | {timestamps[i]:.3f}s | transition {start:g}-{end:g} | sigma {params["sigma_px"]:g}px',x+8,y+h+53,.40)
            frame=av.VideoFrame.from_ndarray(canvas,format='rgb24')
            frame.pts=int(round((timestamps[i]-timestamps[0])*1e6));frame.time_base=time_base
            for packet in stream.encode(frame):container.mux(packet)
            if i in selected:previews.append(canvas)
            row={'frame':i,'timestamp_s':timestamps[i]}
            for j,name in enumerate(('DA','E4')):
                row[name+'_previous_mean_blur_weight']=float(weights[j].mean())
                row[name+'_adjusted_mean_blur_weight']=float(weights[j+2].mean())
                row[name+'_mean_absolute_rgb_settings_change']=float(np.abs(effects[j+2].astype(float)-effects[j]).mean())
            rows.append(row)
        for packet in stream.encode():container.mux(packet)
    Image.fromarray(np.concatenate(previews)).save(output/'settings_preview.png')
    write_csv(output/'settings_comparison.csv',rows)
    with av.open(str(output/'settings_comparison.mp4')) as container:
        decoded=[float(f.pts*f.time_base) for f in container.decode(video=0)]
    if len(decoded)!=len(timestamps) or not np.allclose(decoded,np.array(timestamps)-timestamps[0],atol=2e-6,rtol=0):
        raise ValueError('Settings comparison changed frame count/timestamps')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','baseline','improved'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--output-dir')
    p.add_argument('--transition',type=float,nargs=2,default=(.45,.70),metavar=('START','END'),
                   help='Normalized near score: fully blurred below START, sharp above END')
    p.add_argument('--sigma-px',type=float,default=5.,help='Gaussian blur strength in cache pixels')
    p.add_argument('--reference-application',help='Preserved old application; also render a matched old/new settings comparison')
    a=p.parse_args();cache=repo_path(a.cache)
    if (not np.isfinite([*a.transition,a.sigma_px]).all() or a.sigma_px<=0
            or not 0<=a.transition[0]<a.transition[1]<=1):
        p.error('Require 0 <= START < END <= 1 and positive finite sigma-px')
    custom=tuple(a.transition)!=(.45,.70) or a.sigma_px!=5.
    if (custom or a.reference_application) and not a.output_dir:
        p.error('Custom application settings require a new explicit --output-dir to preserve the original')
    cm=json.loads((cache/'metadata.json').read_text(encoding='utf-8'))
    output=clip_output('application',cm,a.output_dir)
    if output.exists():
        raise FileExistsError('Preserve previous application exports; use a new directory')
    cm=json.loads((cache/'metadata.json').read_text(encoding='utf-8'))
    if checksum(cache/'frames.npy')!=cm['frames_sha256'] or checksum(cache/'frames.csv')!=cm['index_sha256']:
        raise ValueError('Application RGB/time cache changed')
    bm,base,bvalid=load_run(repo_path(a.baseline),cm)
    im,improved,ivalid=load_run(repo_path(a.improved),cm)
    if any(m['prediction_type']!='inverse_depth_like' for m in (bm,im)):
        raise ValueError('This application explicitly uses relative inverse-depth-like values')
    if im.get('config_status')!='frozen':
        raise ValueError('Formal application uses a frozen improvement')
    limits=list(map(float,np.quantile(base[bvalid],[.01,.99])))
    if im['normalization_reference']['limits']!=limits:
        raise ValueError('Use exactly the unchanged method normalization')
    if not np.isfinite(limits).all() or limits[1]<=limits[0]:
        raise ValueError('Application requires a nonzero fixed prediction span')
    parameters={'sigma_px':a.sigma_px,'transition_normalized':list(a.transition),'limits':limits}
    methods=[{'name':m['model']['model'],'metadata_sha256':checksum(repo_path(path)/'metadata.json'),
              'provisional':m.get('config_status')=='provisional'} for m,path in [(bm,a.baseline),(im,a.improved)]]
    reference=None
    if a.reference_application:
        reference=checked_reference(repo_path(a.reference_application),checksum(cache/'metadata.json'),
            {m['name']:m['metadata_sha256'] for m in methods},limits)
    frames=np.load(cache/'frames.npy',mmap_mode='r',allow_pickle=False)
    with (cache/'frames.csv').open(encoding='utf-8',newline='') as f:
        timestamps=[float(r['timestamp_s']) for r in csv.DictReader(f)]
    if base.shape!=frames.shape[:3] or improved.shape!=base.shape or len(timestamps)!=len(frames):
        raise ValueError('Application inputs must have identical frames/output sizes')
    output.mkdir(parents=True)
    h,w=frames.shape[1:3];height=h+72;width=w*4
    previews=[];rows=[];selected={0,len(frames)//2,len(frames)-1}
    time_base=Fraction(1,1000000)
    with av.open(str(output/'demo.mp4'),'w') as container:
        stream=container.add_stream('libx264',rate=Fraction(str(cm['requested_fps'])))
        stream.width,stream.height,stream.pix_fmt=width,height,'yuv420p'
        stream.time_base=time_base;stream.codec_context.time_base=time_base
        stream.options={'crf':'19','preset':'fast'}
        for i,rgb in enumerate(frames):
            original_effect,original_weight=background_blur(rgb,base[i],bvalid[i],limits,sigma=a.sigma_px,transition=a.transition)
            effect,weight=background_blur(rgb,improved[i],ivalid[i],limits,sigma=a.sigma_px,transition=a.transition)
            # Independent direct re-evaluation from the raw saved array.
            repeated,repeated_weight=background_blur(rgb,np.asarray(improved[i]).copy(),ivalid[i],limits,sigma=a.sigma_px,transition=a.transition)
            if not np.array_equal(repeated,effect) or not np.array_equal(repeated_weight,weight):
                raise ValueError('Application result is not reproducible from raw predictions')
            panels=[rgb,colorize(improved[i],ivalid[i],*limits),original_effect,effect]
            labels=['RGB','E4 relative depth (not meters)','Background blur: unchanged DA','Background blur: E4 fusion']
            canvas=np.full((height,width,3),24,np.uint8)
            for j,(panel,label) in enumerate(zip(panels,labels)):
                canvas[30:30+h,j*w:(j+1)*w]=panel
                text(canvas,label,j*w+8,21)
                text(canvas,f'frame {i} | {timestamps[i]:.3f}s | fixed whole-clip scale',j*w+8,h+53,.42)
            frame=av.VideoFrame.from_ndarray(canvas,format='rgb24')
            frame.pts=int(round((timestamps[i]-timestamps[0])*1e6));frame.time_base=time_base
            for packet in stream.encode(frame):container.mux(packet)
            if i in selected:previews.append(canvas)
            rows.append({'frame':i,'timestamp_s':timestamps[i],
                         'baseline_mean_blur_weight':float(original_weight.mean()),
                         'improved_mean_blur_weight':float(weight.mean()),
                         'mean_absolute_rgb_effect_change':float(np.abs(effect.astype(float)-original_effect).mean())})
        for packet in stream.encode():container.mux(packet)
    Image.fromarray(np.concatenate(previews)).save(output/'preview.png')
    write_csv(output/'blur_strength.csv',rows)
    with av.open(str(output/'demo.mp4')) as container:
        decoded=[float(f.pts*f.time_base) for f in container.decode(video=0)]
    if len(decoded)!=len(timestamps) or not np.allclose(decoded,np.array(timestamps)-timestamps[0],atol=2e-6,rtol=0):
        raise ValueError('Application export changed frame count/timestamps')
    files=['demo.mp4','preview.png','blur_strength.csv']
    comparison=None
    if reference is not None:
        settings_comparison(frames,timestamps,base,bvalid,improved,ivalid,limits,reference['parameters'],parameters,output,cm['requested_fps'])
        files+=['settings_comparison.mp4','settings_preview.png','settings_comparison.csv']
        comparison={'reference_application_metadata_sha256':checksum(repo_path(a.reference_application)/'metadata.json'),
                    'reference_application_path':str(repo_path(a.reference_application)),
                    'previous_parameters':reference['parameters'],'adjusted_parameters':parameters,
                    'rows':['previous settings','adjusted settings'],'columns':['unchanged DA','frozen E4'],
                    'same_predictions_and_fixed_scale':True,'decoded_timestamps_match':True,
                    'interpretation':'Application settings exploration after inspecting the clip, not a new blind depth-model improvement result; RGB differences are not quality scores.'}
    write_json(output/'metadata.json',{
        'status':'complete','scope':cm['scope'],'clip_id':cm['clip_id'],'frames':len(frames),
        'application':'relative_background_blur','cache_metadata_sha256':checksum(cache/'metadata.json'),
        'methods':methods,
        'parameters':parameters,
        'settings_comparison':comparison,
        'settings_context':'exploratory_custom_settings' if custom or reference is not None else 'original_settings',
        'verification':{'decoded_timestamps_match':True,'application_recomputed_from_raw':True},
        'files':{name:checksum(output/name) for name in files},
        'implementation_sha256':{name:checksum(ROOT/name) for name in ['scripts/application/render_blur_demo.py','src/p3_depth/application.py']},
        'interpretation':'Relative depth-based artistic effect, not calibrated distance or physical lens simulation; uncertain boundaries can bleed',
    })
    print(f'PASS: {len(frames)} RGB/depth/blur frames; raw-array recomputation and timestamps verified')


if __name__=='__main__':main()
