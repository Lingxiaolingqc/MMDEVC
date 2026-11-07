"""Automatic previews from saved raw predictions; model timings exclude this work."""
import csv
from fractions import Fraction
import html
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from _common import ROOT, checksum, repo_path, write_json


def checked_comparison(cache, runs, output):
    from render_depth_video import render_comparison
    if not output.exists():
        render_comparison(cache, runs, output)
        return
    meta=json.loads((output/'metadata.json').read_text(encoding='utf-8'))
    cm=json.loads((cache/'metadata.json').read_text(encoding='utf-8'))
    for name,key,render_key in [('frames.npy','frames_sha256','cache_frames_sha256'),
                                ('frames.csv','index_sha256','cache_index_sha256')]:
        if checksum(cache/name)!=cm[key] or meta[render_key]!=cm[key]:
            raise ValueError('Existing visualization cache changed')
    if len(meta['inputs'])!=len(runs):
        raise ValueError('Existing visualization uses different methods')
    for run,saved in zip(runs,meta['inputs']):
        if saved['metadata_sha256']!=checksum(run/'metadata.json'):
            raise ValueError('Existing visualization belongs to different inputs')
        rm=json.loads((run/'metadata.json').read_text(encoding='utf-8'))
        for name,key in [('depths.npy','depths_sha256'),('valid.npy','valid_sha256')]:
            if checksum(run/name)!=rm[key]:
                raise ValueError('Prediction artifact changed')
    for name,key in [('comparison.mp4','video_sha256'),('preview.png','preview_sha256')]:
        if checksum(output/name)!=meta[key]:
            raise ValueError('Existing visualization changed')


def video_preview(cache, run):
    output = run / 'visualizations'
    checked_comparison(cache,[run],output)
    return output


def fusion_previews(cache, suite, baseline, vda):
    meta=json.loads((suite/'metadata.json').read_text(encoding='utf-8'))
    if meta['status']!='complete' or any(checksum(path/'metadata.json')!=meta[key] for path,key in
            [(cache,'cache_metadata_sha256'),(baseline,'base_metadata_sha256'),(vda,'vda_metadata_sha256')]):
        raise ValueError('Use the completed suite and its original comparison inputs')
    for variant in ('E0','E1','E2','E3','E4'):
        video_preview(cache,suite/variant)
    for name,runs in [('primary',[baseline,suite/'E4',vda]),
                      ('ablations',[suite/f'E{i}' for i in range(5)])]:
        output=suite/'visualizations'/name
        checked_comparison(cache,runs,output)


def image_preview(rgb, raw, gt, output, label, relative=False):
    from render_depth_video import colorize
    valid = np.isfinite(raw)
    arrays = [rgb]; labels = ['RGB']
    if gt is not None:
        arrays.append(colorize(gt, np.isfinite(gt)&(gt>0), 0, 10))
        labels.append('GT depth | 0-10 m')
    if relative:
        finite = raw[valid]
        low, high = np.quantile(finite, [.01,.99]) if len(finite) else (0,1)
        high = max(float(high), float(low)+1e-6)
        arrays.append(colorize(raw, valid, low, high)); labels.append('Relative depth | not meters')
    else:
        arrays.append(colorize(raw, valid&(raw>0), 0, 10)); labels.append('Prediction | 0-10 m')
        if gt is not None:
            from p3_depth.data.nyuv2 import evaluation_mask
            region = evaluation_mask(gt) & valid
            error = np.abs(np.clip(raw,.001,10)-gt)
            arrays.append(colorize(error, region, 0, 1)); labels.append('Course absolute error | 0-1 m')
    h,w = rgb.shape[:2]
    try:
        font=ImageFont.load_default(size=18)
    except TypeError:  # Pillow 10.0 has no size argument.
        font=ImageFont.load_default()
    canvas = Image.new('RGB',(w*len(arrays),h+52),(24,24,24)); draw=ImageDraw.Draw(canvas)
    for i,(array,title) in enumerate(zip(arrays,labels)):
        canvas.paste(Image.fromarray(array),(i*w,52))
        draw.text((i*w+8,6),label,fill='white',font=font)
        draw.text((i*w+8,29),title,fill='white',font=font)
    output.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(output)


def nyu_gallery(run, rows, data_root, model_name):
    """Render all saved samples without loading a model, preserving every raw byte."""
    folder=run/'visualizations'; folder.mkdir(exist_ok=True)
    entries=[]; relative=model_name=='dav2_relative'
    for row in rows:
        sample=row['sample_id']; pred=run/'predictions'/f'{sample}.npy'
        record=json.loads((run/'records'/f'{sample}.json').read_text(encoding='utf-8'))
        rgb_path=(data_root/row['rgb_path']).resolve(); gt_path=(data_root/row['depth_gt_path']).resolve()
        if not rgb_path.is_relative_to(data_root) or not gt_path.is_relative_to(data_root):
            raise ValueError('Image path escaped data root')
        for path,key in [(pred,'prediction_sha256'),(rgb_path,'rgb_sha256'),(gt_path,'gt_sha256')]:
            if checksum(path)!=record[key]: raise ValueError('Image preview source changed: '+sample)
        path=folder/f'{sample}.png'
        with Image.open(rgb_path) as im: rgb=np.array(im.convert('RGB'))
        image_preview(rgb,np.load(pred,allow_pickle=False),np.load(gt_path,allow_pickle=False),path,sample,relative)
        entries.append({'sample_id':sample,'file':path.name,'sha256':checksum(path),'prediction_sha256':record['prediction_sha256']})
    cards='\n'.join(f'<figure><figcaption>{html.escape(e["sample_id"])}</figcaption><a href="{e["file"]}"><img loading="lazy" src="{e["file"]}" width="960"></a></figure>' for e in entries)
    (folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>NYUv2 predictions</title><h1>NYUv2 predictions</h1>'+cards,encoding='utf-8')
    write_json(folder/'metadata.json',{'images':len(entries),'raw_predictions_modified':False,
        'prediction_type':'relative' if relative else 'metric_depth','display_depth_m':[0,10] if not relative else None,
        'error_display_m':[0,1] if not relative else None,'files':entries,
        'implementation_sha256':checksum(ROOT/'scripts/_media.py')})
    print(f'PASS: {len(entries)} individual PNG previews and gallery: {folder}',flush=True)


def flow_preview(cache, run):
    import av
    import cv2
    meta=json.loads((run/'metadata.json').read_text(encoding='utf-8'))
    cm=json.loads((cache/'metadata.json').read_text(encoding='utf-8'))
    if meta['cache_metadata'] != cm: raise ValueError('Flow cache identity changed')
    for name,key in [('frames.npy','frames_sha256'),('frames.csv','index_sha256')]:
        if checksum(cache/name)!=cm[key]: raise ValueError('Video cache changed')
    for name,digest in meta['artifacts_sha256'].items():
        if checksum(run/(name+'.npy'))!=digest: raise ValueError('Flow data changed')
    output=run/'visualizations'
    if output.exists():
        saved=json.loads((output/'metadata.json').read_text(encoding='utf-8'))
        if saved['flow_metadata_sha256']!=checksum(run/'metadata.json'):
            raise ValueError('Existing flow visualization uses different inputs')
        for name,key in [('flow.mp4','video_sha256'),('preview.png','preview_sha256')]:
            if checksum(output/name)!=saved[key]:
                raise ValueError('Existing flow visualization changed')
        return output
    output.mkdir()
    rgb=np.load(cache/'frames.npy',mmap_mode='r',allow_pickle=False)
    flow=np.load(run/'backward.npy',mmap_mode='r',allow_pickle=False)
    masks=np.load(run/'common_mask.npy',mmap_mode='r',allow_pickle=False)
    with (cache/'frames.csv').open(encoding='utf-8',newline='') as f:
        times=[float(r['timestamp_s']) for r in csv.DictReader(f)]
    scale=max(float(np.quantile(np.linalg.norm(flow,axis=-1),.99)),1e-6)
    h,w=rgb.shape[1:3]; width=max(w,320); previews=[]
    with av.open(str(output/'flow.mp4'),'w') as c:
        stream=c.add_stream('libx264',rate=Fraction(str(cm['requested_fps'])))
        stream.width=width*3; stream.height=h+48; stream.pix_fmt='yuv420p'
        stream.time_base=stream.codec_context.time_base=Fraction(1,1000000)
        stream.options={'crf':'19','preset':'fast'}
        for t in range(len(rgb)):
            flow_rgb=np.full_like(rgb[t],180); mask_rgb=np.full_like(rgb[t],180)
            if t:
                mag,angle=cv2.cartToPolar(flow[t-1,:,:,0],flow[t-1,:,:,1],angleInDegrees=True)
                hsv=np.zeros_like(rgb[t]); hsv[:,:,0]=np.mod(angle/2,180).astype(np.uint8)
                hsv[:,:,1]=255; hsv[:,:,2]=np.rint(np.clip(mag/scale,0,1)*255).astype(np.uint8)
                flow_rgb=cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
                mask_rgb=np.repeat((masks[t-1].astype(np.uint8)*255)[:,:,None],3,axis=2)
            canvas=Image.new('RGB',(width*3,h+48),(24,24,24)); draw=ImageDraw.Draw(canvas)
            for i,(array,title) in enumerate(zip([rgb[t],flow_rgb,mask_rgb],['RGB','Backward flow | fixed magnitude scale','Common reliable mask'])):
                canvas.paste(Image.fromarray(array),(i*width,48));draw.text((i*width+5,8),title,fill='white')
                draw.text((i*width+5,25),f'Frame {t} | {times[t]:.3f} s',fill='white')
            frame=av.VideoFrame.from_ndarray(np.array(canvas),format='rgb24')
            frame.pts=round((times[t]-times[0])*1000000);frame.time_base=Fraction(1,1000000)
            for packet in stream.encode(frame): c.mux(packet)
            if t in {0,len(rgb)//2,len(rgb)-1}: previews.append(np.array(canvas))
        for packet in stream.encode():c.mux(packet)
    with av.open(str(output/'flow.mp4')) as c:
        decoded=[float(f.pts*f.time_base) for f in c.decode(video=0)]
    if len(decoded)!=len(times) or not np.allclose(decoded,np.array(times)-times[0],rtol=0,atol=2e-6):
        raise ValueError('Flow visualization changed timestamps')
    Image.fromarray(np.concatenate(previews)).save(output/'preview.png')
    write_json(output/'metadata.json',{'frames':len(times),'magnitude_limit_px':scale,'first_frame':'no previous frame; gray',
        'flow_metadata_sha256':checksum(run/'metadata.json'),'video_sha256':checksum(output/'flow.mp4'),
        'preview_sha256':checksum(output/'preview.png'),'raw_arrays_modified':False})
    print(f'PASS: flow video and preview: {output}',flush=True)
