"""Fixed whole-clip normalization and synchronized comparison renders."""
import math
from pathlib import Path

import cv2
import numpy as np


def compute_display_range(raw_sequence):
    values=np.asarray(raw_sequence)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Display normalization requires nonempty finite values")
    return tuple(float(v) for v in np.percentile(values,[2,98]))


def normalize_depth(depth,display_range):
    lo,hi=display_range
    if not np.isfinite(depth).all() or not np.isfinite([lo,hi]).all():
        raise ValueError("Depth/display range must be finite")
    return np.clip((np.asarray(depth,dtype=np.float32)-lo)/max(hi-lo,1e-6),0,1)


def colorize(depth,display_range):
    values=np.rint(255*normalize_depth(depth,display_range)).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(values,cv2.COLORMAP_INFERNO),cv2.COLOR_BGR2RGB)


def render_comparison(frame_ids,methods,ranges,fps,output):
    if not frame_ids or fps<=0 or not methods:
        raise ValueError("Comparison requires frames, methods, and positive FPS")
    if any(b!=a+1 for a,b in zip(frame_ids,frame_ids[1:])):
        raise ValueError("Comparison video frame IDs must be consecutive")
    if any(len(values)!=len(frame_ids) for values in methods.values()):
        raise ValueError("All comparison methods must have the same frame count")
    shape=next(iter(methods.values())).shape[1:3]
    height,width=shape
    columns=min(2,len(methods)); rows=math.ceil(len(methods)/columns)
    header=28
    canvas_h=rows*(height+header); canvas_w=columns*width
    canvas_h+=canvas_h%2; canvas_w+=canvas_w%2
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),float(fps),(canvas_w,canvas_h))
    if not writer.isOpened():
        raise OSError(f"Cannot create comparison video {output}")
    try:
        for index,frame_id in enumerate(frame_ids):
            canvas=np.zeros((canvas_h,canvas_w,3),np.uint8)
            for position,(name,values) in enumerate(methods.items()):
                image=values[index]
                if image.shape[:2]!=shape:
                    raise ValueError("Comparison outputs must share an output canvas size")
                rgb=colorize(image,ranges[name]) if image.ndim==2 else image
                row,column=divmod(position,columns)
                top=row*(height+header); left=column*width
                canvas[top+header:top+header+height,left:left+width]=rgb
                cv2.putText(canvas,f'{name} | frame {frame_id}',(left+8,top+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(240,240,240),1,cv2.LINE_AA)
            writer.write(cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
