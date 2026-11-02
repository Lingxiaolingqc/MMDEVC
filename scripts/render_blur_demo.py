"""Export RGB, relative depth and matched baseline/improved background-blur effects."""

import argparse
import csv
from fractions import Fraction
import json
import av
import numpy as np
from PIL import Image

from run_stereo_baseline import ROOT,repo_path,checksum,write_json,write_csv
from run_fusion_suite import load_run
from render_depth_video import colorize,text
from p3_depth.application import background_blur


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','baseline','improved','output-dir'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args();cache,output=repo_path(a.cache),repo_path(a.output_dir)
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
            original_effect,original_weight=background_blur(rgb,base[i],bvalid[i],limits)
            effect,weight=background_blur(rgb,improved[i],ivalid[i],limits)
            # Independent direct re-evaluation from the raw saved array.
            repeated,repeated_weight=background_blur(rgb,np.asarray(improved[i]).copy(),ivalid[i],limits)
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
    write_json(output/'metadata.json',{
        'status':'complete','scope':cm['scope'],'clip_id':cm['clip_id'],'frames':len(frames),
        'application':'relative_background_blur','cache_metadata_sha256':checksum(cache/'metadata.json'),
        'methods':[{'name':m['model']['model'],'metadata_sha256':checksum(repo_path(path)/'metadata.json'),
                    'provisional':m.get('config_status')=='provisional'} for m,path in [(bm,a.baseline),(im,a.improved)]],
        'parameters':{'sigma_px':5.,'transition_normalized':[.45,.70],'limits':limits},
        'verification':{'decoded_timestamps_match':True,'application_recomputed_from_raw':True},
        'files':{name:checksum(output/name) for name in ('demo.mp4','preview.png','blur_strength.csv')},
        'implementation_sha256':{name:checksum(ROOT/name) for name in ['scripts/render_blur_demo.py','src/p3_depth/application.py']},
        'interpretation':'Relative depth-based artistic effect, not calibrated distance or physical lens simulation; uncertain boundaries can bleed',
    })
    print(f'PASS: {len(frames)} RGB/depth/blur frames; raw-array recomputation and timestamps verified')


if __name__=='__main__':main()
