"""A qualitative near-sharp / far-blurred effect from raw relative inverse depth."""

import cv2
import numpy as np


def background_blur(rgb, relative, valid, limits, sigma=5., transition=(.45,.70)):
    if (rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3
            or relative.shape!=rgb.shape[:2] or valid.shape!=relative.shape or valid.dtype!=np.bool_):
        raise ValueError('Expected matching uint8 RGB, relative HW prediction and boolean mask')
    low,high=map(float,limits);start,end=transition
    if not np.isfinite([low,high,sigma,start,end]).all() or high<=low or sigma<=0 or not 0<=start<end<=1:
        raise ValueError('Require fixed finite display limits and valid blur parameters')
    usable=valid & np.isfinite(relative)
    near=np.clip((np.where(usable,relative,high)-low)/(high-low),0,1)
    t=np.clip((near-start)/(end-start),0,1)
    weight=np.where(usable,1-t*t*(3-2*t),0).astype(np.float32)
    blurred=cv2.GaussianBlur(rgb,(0,0),sigmaX=sigma,sigmaY=sigma)
    result=np.rint(rgb.astype(np.float32)*(1-weight[...,None])+blurred.astype(np.float32)*weight[...,None])
    return np.clip(result,0,255).astype(np.uint8),weight
