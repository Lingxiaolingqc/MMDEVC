"""RGB-only Farneback correspondences and a method-independent evaluation region."""

import cv2
import numpy as np


def sample_previous(values, backward, valid=None):
    """Gather previous-frame values at (x,y)+current-to-previous flow.

    Require all interpolation support to be valid; never turn invalid depth into 0.
    OpenCV INTER_LINEAR uses its documented fractional interpolation table.
    """
    values = np.asarray(values, dtype=np.float32)
    backward = np.asarray(backward, dtype=np.float32)
    h,w = values.shape[:2]
    if values.ndim not in (2,3) or backward.shape != (h,w,2):
        raise ValueError('Expected HW or HWC values and HW2 backward flow')
    finite = np.isfinite(values) if values.ndim==2 else np.isfinite(values).all(axis=2)
    if valid is not None:
        if valid.shape!=(h,w) or valid.dtype!=np.bool_:
            raise ValueError('Valid mask must be boolean HW')
        finite &= valid
    y,x = np.indices((h,w),dtype=np.float32)
    qx,qy = x+backward[...,0],y+backward[...,1]
    inside = np.isfinite(backward).all(axis=2) & (qx>=0) & (qx<=w-1) & (qy>=0) & (qy<=h-1)
    qx,qy = np.where(inside,qx,-1),np.where(inside,qy,-1)
    clean = np.where(finite if values.ndim==2 else finite[...,None],values,0)
    result = cv2.remap(clean,qx,qy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    support = cv2.remap(finite.astype(np.float32),qx,qy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    ok = inside & (support>=1-1e-6)
    result[~ok] = np.nan
    return result,ok


def correspondence(previous_rgb, current_rgb, forward, backward, fb_relative=.01, fb_absolute=.5, photo_max=.2):
    """All fields live on current-frame pixels; masks use RGB and flow only."""
    if fb_relative<0 or fb_absolute<0 or not 0<=photo_max<=1:
        raise ValueError('Invalid consistency thresholds')
    sampled_forward,inside = sample_previous(forward,backward)
    warped_rgb,photo_ok = sample_previous(previous_rgb,backward)
    squared_residual = np.sum((backward+sampled_forward)**2,axis=2)
    squared_size = np.sum(backward**2+sampled_forward**2,axis=2)
    residual = np.where(inside,np.sqrt(squared_residual),np.inf).astype(np.float32)
    photo = np.where(photo_ok,np.mean(np.abs(current_rgb.astype(np.float32)-warped_rgb),axis=2)/255.,np.inf).astype(np.float32)
    consistent = inside & photo_ok & (squared_residual <= fb_relative*squared_size+fb_absolute)
    return {'fb_residual':residual,'photo_residual':photo,'in_bounds':inside,
            'common_mask':consistent & (photo<=photo_max)}


def estimate_pair(previous_rgb,current_rgb,settings):
    if previous_rgb.shape!=current_rgb.shape or previous_rgb.ndim!=3 or previous_rgb.shape[2]!=3 or previous_rgb.dtype!=np.uint8 or current_rgb.dtype!=np.uint8:
        raise ValueError('Expected matching uint8 RGB frames')
    previous,current = (cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY) for rgb in (previous_rgb,current_rgb))
    opts = settings['farneback']
    forward = cv2.calcOpticalFlowFarneback(previous,current,None,**opts)
    backward = cv2.calcOpticalFlowFarneback(current,previous,None,**opts)
    return forward,backward,correspondence(previous_rgb,current_rgb,forward,backward,**settings['evaluation_mask'])
