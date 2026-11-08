"""Optical-flow backends and full-resolution displacement conversion."""
import cv2
import numpy as np

from .models import validate_rgb


def resize_flow(flow: np.ndarray, output_hw: tuple[int,int]) -> np.ndarray:
    height,width=output_hw
    source_h,source_w=flow.shape[:2]
    result=cv2.resize(flow,(width,height),interpolation=cv2.INTER_LINEAR)
    result[:,:,0]*=width/source_w
    result[:,:,1]*=height/source_h
    return result.astype(np.float32)


def estimate_pair(prev_rgb: np.ndarray,curr_rgb: np.ndarray,config: dict):
    validate_rgb(prev_rgb); validate_rgb(curr_rgb)
    if prev_rgb.shape!=curr_rgb.shape:
        raise ValueError("Flow frames must have identical shapes")
    if config.get('backend','farneback')!='farneback':
        raise ValueError("Only the verified Farneback backend is currently available")
    height,width=prev_rgb.shape[:2]
    scale=min(1,float(config.get('max_res',640))/max(height,width))
    size=(max(8,round(width*scale)),max(8,round(height*scale)))
    images=[cv2.cvtColor(cv2.resize(image,size,interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2GRAY) for image in [prev_rgb,curr_rgb]]
    options=dict(pyr_scale=0.5,levels=4,winsize=21,iterations=4,poly_n=7,poly_sigma=1.5,flags=0)
    for key in options:
        if key in config:
            options[key]=config[key]
    forward=cv2.calcOpticalFlowFarneback(images[0],images[1],None,**options)
    backward=cv2.calcOpticalFlowFarneback(images[1],images[0],None,**options)
    return resize_flow(forward,(height,width)),resize_flow(backward,(height,width))
