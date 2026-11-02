"""Causal E0-E4 fusion with explicit metric/relative confidence conventions."""

import numpy as np
from p3_depth.temporal.flow import sample_previous


def edge_strength(values,valid):
    strength=np.zeros_like(values,dtype=np.float32)
    for axis in (0,1):
        left,right=[slice(None)]*2,[slice(None)]*2
        left[axis],right[axis]=slice(None,-1),slice(1,None)
        left,right=tuple(left),tuple(right)
        diff=np.where(valid[left]&valid[right],np.abs(values[left]-values[right]),0)
        strength[left]=np.maximum(strength[left],diff)
        strength[right]=np.maximum(strength[right],diff)
    return strength


def log_edge_strength(depth,valid):
    return edge_strength(np.log(np.where(valid,depth,1)).astype(np.float32),valid)


def fuse_step(current,history,variant,parameters,backward=None,forward=None,photo_residual=None,fb_residual=None,representation='metric_depth'):
    if current.ndim!=2 or current.shape!=history.shape or variant not in ('E1','E2','E3','E4'):
        raise ValueError('Expected matching HW predictions and E1-E4 variant')
    if representation not in ('metric_depth','inverse_depth_like'):
        raise ValueError('Unknown depth representation')
    relative=representation=='inverse_depth_like'
    alpha=float(parameters['history_weight'])
    if not 0<=alpha<1:
        raise ValueError('History weight must be in [0,1)')
    current_valid=np.isfinite(current) & (True if relative else current>0)
    history_valid=np.isfinite(history) & (True if relative else history>0)
    if variant=='E1':
        warped,eligible=history,history_valid.copy()
    else:
        if backward is None:
            raise ValueError('Flow variants need current-to-previous flow')
        warped,eligible=sample_previous(history,backward,history_valid)
    eligible &= current_valid
    if variant in ('E3','E4'):
        if any(x is None for x in (forward,photo_residual,fb_residual)):
            raise ValueError('Filtered variants need both flows and RGB/flow residuals')
        sampled_forward,flow_valid=sample_previous(forward,backward)
        norm_squared=np.sum(backward**2+sampled_forward**2,axis=2)
        eligible &= flow_valid & (fb_residual**2<=parameters['fb_relative']*norm_squared+parameters['fb_absolute'])
        eligible &= np.isfinite(photo_residual)&(photo_residual<=parameters['photo_max'])
    weight=np.where(eligible,alpha,0).astype(np.float32)
    if variant=='E4':
        change_key,edge_key=('depth_span_sigma','edge_span_sigma') if relative else ('depth_log_sigma','edge_log_sigma')
        for name in ('photo_sigma','flow_sigma_px',change_key,edge_key):
            if parameters[name]<=0:
                raise ValueError('Confidence scales must be positive')
        if relative:
            # Current-frame span affects confidence only; raw outputs stay unscaled.
            finite=current[current_valid]
            span=float(np.diff(np.quantile(finite,[.01,.99]))[0]) if finite.size else 0.
            if span<=1e-8:
                return np.where(current_valid,current,np.nan).astype(np.float32),np.zeros_like(current,dtype=np.float32)
            change=np.abs(warped-current)/span
            edge=edge_strength(current,current_valid)/span
        else:
            ratio=np.ones_like(current,dtype=np.float32)
            np.divide(warped,current,out=ratio,where=eligible)
            change=np.abs(np.log(ratio))
            edge=log_edge_strength(current,current_valid)
        exponent=(photo_residual/parameters['photo_sigma']+fb_residual/parameters['flow_sigma_px']+
                  change/parameters[change_key]+edge/parameters[edge_key])
        confidence=np.exp(-np.where(eligible,exponent,np.inf))
        weight *= confidence
    # Where current depth is missing, do not invent an estimate from stale history.
    result=np.where(current_valid,current,np.nan).astype(np.float32)
    chosen=weight>0
    result[chosen]=(1-weight[chosen])*current[chosen]+weight[chosen]*warped[chosen]
    return result,weight


def fuse_sequence(depths,valid,variant,parameters,flows=None,representation='metric_depth'):
    if depths.ndim!=3 or valid.shape!=depths.shape or valid.dtype!=np.bool_:
        raise ValueError('Expected THW depth and boolean valid mask')
    if representation not in ('metric_depth','inverse_depth_like'):
        raise ValueError('Unknown depth representation')
    raw=np.where(valid & np.isfinite(depths) & (True if representation=='inverse_depth_like' else depths>0),depths,np.nan).astype(np.float32)
    if variant=='E0':
        return raw,np.zeros_like(raw)
    if variant not in ('E1','E2','E3','E4'):
        raise ValueError('Fusion supports E0-E4; E5 is the independent VDA output')
    result,weights=raw.copy(),np.zeros_like(raw)
    for t in range(1,len(raw)):
        kwargs={} if variant=='E1' else {k:flows[k][t-1] for k in ('backward','forward','photo_residual','fb_residual')}
        result[t],weights[t]=fuse_step(raw[t],result[t-1],variant,parameters,representation=representation,**kwargs)
    return result,weights
