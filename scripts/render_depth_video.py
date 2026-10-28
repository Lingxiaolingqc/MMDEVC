"""Render timestamp-preserving RGB/depth comparisons with clip-wide fixed scales."""

import argparse
import csv
from fractions import Fraction
import json

import av
import cv2
import numpy as np
from PIL import Image

from run_stereo_baseline import ROOT, repo_path, checksum, write_json


def colorize(values, valid, low, high):
    finite = valid & np.isfinite(values)
    scaled = np.clip((np.where(finite, values, low)-low)/(high-low), 0, 1)
    bgr = cv2.applyColorMap(np.rint(scaled*255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb[~finite] = 180
    return rgb


def text(canvas, label, x, y, scale=.45):
    cv2.putText(canvas, label, (x,y), cv2.FONT_HERSHEY_SIMPLEX, scale, (240,240,240), 1, cv2.LINE_AA)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', required=True)
    p.add_argument('--run', action='append', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--depth-max', type=float, default=10., help='Metric display limit only; no prediction clipping')
    args = p.parse_args()
    if not np.isfinite(args.depth_max) or args.depth_max <= 0:
        p.error('Positive finite metric display limit required')
    cache, output = repo_path(args.cache), repo_path(args.output_dir)
    if output.exists():
        raise FileExistsError('Use a new render directory')
    cm = json.loads((cache/'metadata.json').read_text(encoding='utf-8'))
    for name, key in [('frames.npy','frames_sha256'),('frames.csv','index_sha256')]:
        if checksum(cache/name) != cm[key]:
            raise ValueError('Cache hash mismatch')
    frames = np.load(cache/'frames.npy', mmap_mode='r', allow_pickle=False)
    with (cache/'frames.csv').open(encoding='utf-8',newline='') as f:
        times = np.array([float(row['timestamp_s']) for row in csv.DictReader(f)])
    if len(times) != len(frames) or not np.all(np.diff(times)>0):
        raise ValueError('Invalid timestamp index')
    inputs = []
    for name in args.run:
        run = repo_path(name)
        meta = json.loads((run/'metadata.json').read_text(encoding='utf-8'))
        if meta['status'] != 'complete' or any(meta['cache_metadata'][key] != cm[key] for key in ('frames_sha256','index_sha256')):
            raise ValueError('Runs must use exactly this complete RGB/time cache')
        for fname, key in [('depths.npy','depths_sha256'),('valid.npy','valid_sha256')]:
            if checksum(run/fname) != meta[key]:
                raise ValueError('Prediction hash mismatch')
        depth = np.load(run/'depths.npy',mmap_mode='r',allow_pickle=False)
        valid = np.load(run/'valid.npy',mmap_mode='r',allow_pickle=False)
        if depth.shape != frames.shape[:3] or valid.shape != depth.shape:
            raise ValueError('Prediction/cache shapes differ')
        relative = meta['prediction_type'] == 'inverse_depth_like'
        if relative:
            values = depth[valid & np.isfinite(depth)]
            if not len(values):
                raise ValueError('Relative prediction has no finite pixels to set a display scale')
            low, high = map(float,np.quantile(values,[.01,.99]))
            high = max(high,low+1e-6)
        else:
            low, high = 0., args.depth_max
        inputs.append((depth,valid,{'run':name,'model':meta['model']['model'],
            'low':low,'high':high,'unit':'arbitrary inverse' if relative else 'm',
            'scale_rule':'whole-clip 1st/99th percentiles' if relative else 'fixed user metric range',
            'metadata_sha256':checksum(run/'metadata.json')}))
    output.mkdir(parents=True)
    h,w = frames.shape[1:3]
    panel_width = max(320,w)
    height,width = h+104,panel_width*(len(inputs)+1)
    if height%2 or width%2:
        raise ValueError('Expected even cache dimensions')
    previews = []
    preview_ids = {0,len(frames)//2,len(frames)-1}
    time_base = Fraction(1,1000000)
    with av.open(str(output/'comparison.mp4'),'w') as container:
        stream = container.add_stream('libx264',rate=Fraction(str(cm['requested_fps'])))
        stream.width,stream.height,stream.pix_fmt = width,height,'yuv420p'
        stream.time_base = time_base
        stream.codec_context.time_base = time_base
        stream.options = {'crf':'19','preset':'fast'}
        for i,rgb in enumerate(frames):
            canvas = np.full((height,width,3),24,dtype=np.uint8)
            canvas[40:40+h,:w] = rgb
            text(canvas,f'RGB | t={times[i]:.3f}s | frame {i}',8,22)
            text(canvas,'Gray = unavailable / invalid',8,h+61)
            text(canvas,'Display limits fixed for entire clip',8,h+83,.40)
            for j,(depth,valid,info) in enumerate(inputs,1):
                x=j*panel_width
                canvas[40:40+h,x:x+w]=colorize(depth[i],valid[i],info['low'],info['high'])
                text(canvas,info['model'],x+8,22)
                ramp=np.linspace(info['low'],info['high'],panel_width-16)[None,:].repeat(10,axis=0)
                canvas[h+49:h+59,x+8:x+panel_width-8]=colorize(ramp,np.ones_like(ramp,dtype=bool),info['low'],info['high'])
                text(canvas,f"{info['low']:.2f} ... {info['high']:.2f} {info['unit']}",x+8,h+80,.40)
            frame=av.VideoFrame.from_ndarray(canvas,format='rgb24')
            frame.pts=int(round((times[i]-times[0])*1000000))
            frame.time_base=time_base
            for packet in stream.encode(frame):
                container.mux(packet)
            if i in preview_ids:
                previews.append(canvas)
        for packet in stream.encode():
            container.mux(packet)
    Image.fromarray(np.concatenate(previews)).save(output/'preview.png')
    with av.open(str(output/'comparison.mp4')) as container:
        decoded_times = [float(f.pts*f.time_base) for f in container.decode(video=0)]
    if len(decoded_times)!=len(times) or not np.allclose(decoded_times,times-times[0],rtol=0,atol=2e-6):
        raise ValueError('Rendered video did not preserve frame count/timestamps')
    write_json(output/'metadata.json',{'status':'complete','scope':cm['scope'],'frames':len(times),
        'cache_frames_sha256':cm['frames_sha256'],'cache_index_sha256':cm['index_sha256'],
        'inputs':[item[2] for item in inputs],'raw_predictions_modified':False,
        'timing':'original selected timestamps retained; verified by decoding rendered MP4',
        'video_sha256':checksum(output/'comparison.mp4'),'preview_sha256':checksum(output/'preview.png'),
        'renderer_sha256':checksum(ROOT/'scripts/render_depth_video.py')})
    print(f'PASS: {len(times)} rendered frames with original timestamps and fixed scales: {output}')


if __name__ == '__main__':
    main()
