"""GPU-synchronized full-pipeline timing, separate from cached replay."""
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .flow import estimate_pair
from .fusion import update
from .models import load_image_adapter
from .records import sha256_file,run_provenance
from .video import inspect_video,iter_rgb_frames


def benchmark_frames(frames,adapter,config,repeats=3,torch_module=None):
    if not frames or repeats<1:
        raise ValueError("Benchmark requires frames and positive repeats")
    gpu=torch_module is not None and torch_module.cuda.is_available()
    sync=lambda:torch_module.cuda.synchronize() if gpu else None
    for _ in range(2): adapter.predict_rgb(frames[0].rgb)
    sync()
    rows=[]
    for trial in range(repeats):
        state=None
        if gpu: torch_module.cuda.reset_peak_memory_stats()
        depth_s=0.;flow_s=0.;fusion_s=0.
        sync();start=time.perf_counter()
        for frame in frames:
            sync();t=time.perf_counter();prediction=adapter.predict_rgb(frame.rgb);sync()
            depth_s+=time.perf_counter()-t
            pair=None
            if state is not None and config.get('method','adaptive') not in {'baseline','ema'}:
                t=time.perf_counter();pair=estimate_pair(state.rgb,frame.rgb,config.get('flow',{}))
                flow_s+=time.perf_counter()-t
            t=time.perf_counter();_,state,_=update(frame,prediction,state,config,flows=pair)
            fusion_s+=time.perf_counter()-t
        sync();elapsed=time.perf_counter()-start
        row={'trial':trial+1,'frame_count':len(frames),'pipeline_s':elapsed,'fps':len(frames)/elapsed,'depth_s':depth_s,'flow_s':flow_s,'fusion_s':fusion_s,'peak_allocated_mb':torch_module.cuda.max_memory_allocated()/1024**2 if gpu else None,'peak_reserved_mb':torch_module.cuda.max_memory_reserved()/1024**2 if gpu else None}
        rows.append(row)
        print(f"BENCH trial {trial+1}: {row['fps']:.2f} FPS | depth {depth_s:.3f}s flow {flow_s:.3f}s fusion {fusion_s:.3f}s",flush=True)
    return {'trials':rows,'median_fps':float(np.median([row['fps'] for row in rows])),'min_fps':float(min(row['fps'] for row in rows)),'max_fps':float(max(row['fps'] for row in rows)),'median_frame_s':float(np.median([row['pipeline_s']/len(frames) for row in rows])),'scope':'full depth inference + both optical-flow directions + fusion; RGB decode/resize and disk video encoding excluded; no cached depth or flow','warmup_calls':2,'repeats':repeats}


def benchmark_pipeline(config,repeats=3):
    import torch

    root=Path(config['root']);source=Path(config['source'])
    provenance=run_provenance(root)
    params=json.loads(Path(config['fusion_config']).read_text())
    info=inspect_video(source)
    frames=list(iter_rgb_frames(source,100))
    height,width=frames[0].rgb.shape[:2]
    scale=min(1,640/max(height,width))
    for frame in frames:
        frame.rgb=cv2.resize(frame.rgb,(round(width*scale),round(height*scale)),interpolation=cv2.INTER_AREA)
    torch.set_num_threads(4)
    adapter=load_image_adapter('dav2',root,518,'fp16')
    result=benchmark_frames(frames,adapter,params,repeats,torch)
    result['provenance']=provenance
    result.update(source_sha256=info.source_sha256,configuration_sha256=sha256_file(config['fusion_config']),hardware=torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU',input_size=518,output_hw=list(frames[0].rgb.shape[:2]),precision='fp16 depth; fp32 flow/fusion',source_compliant=info.compliant)
    return result
