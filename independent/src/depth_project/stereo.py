"""Classical stereo in pixel disparity and calibrated meters."""
import cv2
import numpy as np


def fixed_point_to_pixels(raw: np.ndarray) -> np.ndarray:
    return np.asarray(raw,dtype=np.float32)/16


def disparity_to_meters(disp_px: np.ndarray, focal_px: float, baseline_mm: float, doffs: float) -> np.ndarray:
    disparity=np.asarray(disp_px,dtype=np.float64)
    denominator=disparity+doffs
    valid=np.isfinite(disparity)&(disparity>0)&(denominator>0)
    depth=np.full(disparity.shape,np.nan,dtype=np.float64)
    depth[valid]=focal_px*baseline_mm/denominator[valid]/1000
    return depth


def compute_sgbm(left: np.ndarray,right: np.ndarray,num_disparities: int=256,block_size: int=5) -> np.ndarray:
    if left.shape!=right.shape or left.ndim!=2:
        raise ValueError("Stereo requires equal-size grayscale rectified images")
    if num_disparities<=0 or num_disparities%16 or block_size<3 or block_size%2==0:
        raise ValueError("num_disparities must be a positive multiple of 16; block_size odd >=3")
    matcher=cv2.StereoSGBM_create(minDisparity=0,numDisparities=num_disparities,blockSize=block_size,P1=8*block_size**2,P2=32*block_size**2,disp12MaxDiff=1,uniquenessRatio=10,speckleWindowSize=100,speckleRange=2,preFilterCap=31,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    return fixed_point_to_pixels(matcher.compute(left,right))
