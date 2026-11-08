"""Course-protocol metrics and explicit diagnostic scores."""
import cv2
import numpy as np


def depth_metrics(pred_valid: np.ndarray, gt_valid: np.ndarray) -> dict:
    pred=np.asarray(pred_valid,dtype=np.float64)
    gt=np.asarray(gt_valid,dtype=np.float64)
    if pred.shape!=gt.shape or pred.size==0:
        raise ValueError("Depth metrics require matching nonempty valid vectors")
    if not np.isfinite(pred).all() or not np.isfinite(gt).all() or (pred<=0).any() or (gt<=0).any():
        raise ValueError("Depth metrics require finite positive values")
    error=pred-gt
    return {'abs_rel':float(np.mean(np.abs(error)/gt)), 'rmse':float(np.sqrt(np.mean(error**2))), 'delta1':float(np.mean(np.maximum(pred/gt,gt/pred)<1.25)), 'valid_count':int(pred.size)}


def evaluate_nyu(pred_m: np.ndarray, gt_m: np.ndarray) -> dict:
    pred=np.asarray(pred_m,dtype=np.float64)
    gt=np.asarray(gt_m,dtype=np.float64)
    if pred.ndim!=2 or gt.shape!=(480,640):
        raise ValueError("NYU expects HW predictions and 480x640 ground truth")
    if not np.isfinite(pred).all():
        raise ValueError("Non-finite model prediction")
    if pred.shape!=gt.shape:
        pred=cv2.resize(pred,(640,480),interpolation=cv2.INTER_LINEAR)
    pred=np.clip(pred,0.001,10)[45:471,41:601]
    gt=gt[45:471,41:601]
    valid=np.isfinite(gt)&(gt>0.001)&(gt<10)
    if not valid.any():
        raise ValueError("NYU image has no valid evaluation pixels")
    return depth_metrics(pred[valid],gt[valid])


def average_image_metrics(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("No per-image metrics")
    return {key:float(np.mean([row[key] for row in rows])) for key in ['abs_rel','rmse','delta1']} | {'image_count':len(rows)}


def stereo_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred=np.asarray(pred); gt=np.asarray(gt)
    if pred.shape!=gt.shape:
        raise ValueError("Stereo prediction/GT shapes differ")
    gt_valid=np.isfinite(gt)&(gt>0)
    supported=gt_valid&np.isfinite(pred)&(pred>0)
    count=int(gt_valid.sum()); predicted=int(supported.sum())
    if not count:
        raise ValueError("Stereo image has no finite positive ground truth")
    return {'mae_px':float(np.mean(np.abs(pred[supported]-gt[supported]))) if predicted else None,'coverage':predicted/count,'gt_valid_count':count,'prediction_valid_count':predicted}
