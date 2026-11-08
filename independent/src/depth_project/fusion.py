"""Causal flow-guided relative-depth fusion with inspectable confidence."""
from dataclasses import dataclass

import numpy as np

from .flow import estimate_pair
from .records import DepthPrediction,Frame
from .warp import flow_consistency,warp_previous


@dataclass(slots=True)
class FusionState:
    rgb: np.ndarray
    depth: np.ndarray
    frame_id: int


def fuse_step(current,aligned_previous,valid,confidence,history_cap):
    current=np.asarray(current,dtype=np.float32)
    previous=np.asarray(aligned_previous,dtype=np.float32)
    valid=np.asarray(valid,dtype=bool)
    confidence=np.asarray(confidence,dtype=np.float32)
    if not 0<=history_cap<=1:
        raise ValueError("History cap must lie in [0,1]")
    if any(a.shape!=current.shape for a in [previous,valid,confidence]):
        raise ValueError("Fusion arrays must have matching HW shape")
    if not np.isfinite(current).all():
        raise ValueError("Current prediction must be finite")
    supported=valid&np.isfinite(previous)
    weights=history_cap*np.clip(np.nan_to_num(confidence,nan=0,posinf=0,neginf=0),0,1)*supported
    safe_previous=np.where(supported,previous,current)
    return ((1-weights)*current+weights*safe_previous).astype(np.float32)


def update(frame: Frame,prediction: DepthPrediction,state: FusionState | None,config: dict,flows=None):
    if prediction.kind!='relative':
        raise ValueError("This fusion pipeline expects relative depth")
    current=prediction.values
    if current.shape!=frame.rgb.shape[:2]:
        raise ValueError("Frame and depth shapes differ")
    if state is not None and frame.index!=state.frame_id+1:
        raise ValueError("Fusion requires consecutive frame IDs")
    method=config.get('method','adaptive')
    if method not in {'baseline','ema','flow','occlusion','adaptive'}:
        raise ValueError(f"Unknown fusion method {method}")
    reset=None
    if state is None:
        reset='first_frame'
    elif state.rgb.shape!=frame.rgb.shape or state.depth.shape!=current.shape:
        raise ValueError("Fusion state shape mismatch")
    elif np.mean(np.abs(frame.rgb.astype(np.float32)-state.rgb.astype(np.float32)))/255>config.get('scene_cut_rgb_difference',0.45):
        reset='scene_cut'
    weights=np.zeros_like(current)
    valid=np.zeros(current.shape,bool)
    if reset or method=='baseline':
        fused=current.copy()
    else:
        cap=float(config.get('history_cap',0.5))
        confidence=np.ones_like(current)
        if method=='ema':
            aligned=state.depth
            valid=np.ones(current.shape,bool)
        else:
            forward,backward=flows if flows is not None else estimate_pair(state.rgb,frame.rgb,config.get('flow',{}))
            aligned,valid=warp_previous(state.depth,backward)
            if method in {'occlusion','adaptive'}:
                valid &= flow_consistency(forward,backward,config.get('consistency',{}))
            if method=='adaptive':
                rgb_aligned,rgb_valid=warp_previous(state.rgb.astype(np.float32)/255,backward)
                valid &= rgb_valid
                photo=np.mean(np.abs(frame.rgb.astype(np.float32)/255-rgb_aligned),axis=2)
                photo_sigma=float(config.get('photo_sigma',0.1))
                depth_sigma=float(config.get('depth_sigma',0.2))
                if photo_sigma<=0 or depth_sigma<=0:
                    raise ValueError("Confidence scales must be positive")
                robust_range=max(float(np.percentile(current,75)-np.percentile(current,25)),1e-6)
                confidence=np.exp(-photo/photo_sigma)*np.exp(-np.abs(current-aligned)/(depth_sigma*robust_range))
        weights=cap*np.clip(confidence,0,1)*valid
        fused=fuse_step(current,aligned,valid,confidence,cap)
    output=DepthPrediction(fused,prediction.kind,prediction.model_id,prediction.checkpoint_sha256,prediction.preprocessing_id,prediction.precision)
    return output,FusionState(frame.rgb.copy(),fused.copy(),frame.index),{'reset':reset,'valid':valid,'history_weight':weights,'mean_weight':float(weights.mean()),'valid_fraction':float(valid.mean())}
