"""Current-to-previous gather warps with explicit support masks."""
import cv2
import numpy as np


def warp_previous(previous: np.ndarray, backward: np.ndarray):
    previous=np.asarray(previous,dtype=np.float32)
    backward=np.asarray(backward,dtype=np.float32)
    height,width=previous.shape[:2]
    if previous.ndim not in {2,3} or backward.shape!=(height,width,2):
        raise ValueError("Warp source and backward flow sizes are incompatible")
    xx,yy=np.meshgrid(np.arange(width,dtype=np.float32),np.arange(height,dtype=np.float32))
    map_x=xx+backward[:,:,0]; map_y=yy+backward[:,:,1]
    valid=np.isfinite(backward).all(axis=2)&(map_x>=0)&(map_x<=width-1)&(map_y>=0)&(map_y<=height-1)
    map_x=np.where(valid,map_x,-1).astype(np.float32)
    map_y=np.where(valid,map_y,-1).astype(np.float32)
    finite=np.isfinite(previous) if previous.ndim==2 else np.isfinite(previous).all(axis=2)
    samples=cv2.remap(np.nan_to_num(previous,nan=0,posinf=0,neginf=0),map_x,map_y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    support=cv2.remap(finite.astype(np.float32),map_x,map_y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    valid &= support>=1-1e-6
    samples[~valid]=0
    return samples,valid


def flow_consistency(forward: np.ndarray, backward: np.ndarray, config: dict) -> np.ndarray:
    sampled,valid=warp_previous(forward,backward)
    residual=np.sum((backward+sampled)**2,axis=2)
    magnitude=np.sum(backward**2+sampled**2,axis=2)
    return valid&(residual<float(config.get('alpha',0.01))*magnitude+float(config.get('beta',0.5)))
