"""Temporal agreement proxies on fixed RGB-only flow regions, never accuracy GT."""

import numpy as np
from p3_depth.temporal.flow import sample_previous


def evaluate_video(depths,valid,backward,common_mask,timestamps,error_cap_m=1.,static_scene=False):
    depths=np.asarray(depths)
    times=np.asarray(timestamps,dtype=np.float64)
    if depths.ndim!=3 or len(depths)<2 or valid.shape!=depths.shape or valid.dtype!=np.bool_:
        raise ValueError('Expected at least two THW depths with boolean mask')
    if backward.shape!=(*depths[1:].shape,2) or common_mask.shape!=depths[1:].shape or common_mask.dtype!=np.bool_:
        raise ValueError('Flow and fixed region must cover every adjacent pair')
    if len(times)!=len(depths) or not np.isfinite(times).all() or not np.all(np.diff(times)>0) or error_cap_m<=0:
        raise ValueError('Strictly increasing timestamps and positive error cap required')
    if np.any(valid & (~np.isfinite(depths)|(depths<=0))):
        raise ValueError('Valid mask includes nonphysical metric depths')
    rows=[]
    for t in range(1,len(depths)):
        previous,previous_valid=sample_previous(depths[t-1],backward[t-1],valid[t-1])
        region=common_mask[t-1]
        chosen=region & valid[t] & previous_valid
        n,k=int(region.sum()),int(chosen.sum())
        current,history=depths[t][chosen].astype(np.float64),previous[chosen].astype(np.float64)
        error=np.abs(current-history)
        relative=error/((current+history)/2)
        dt=float(times[t]-times[t-1])
        rows.append({'frame':t,'timestamp_s':float(times[t]),'dt_s':dt,'reference_pixels':n,'valid_pairs':k,
            'pair_coverage':k/n if n else None,
            'temporal_l1_m_on_valid':float(error.mean()) if k else None,
            'temporal_abs_log_on_valid':float(np.abs(np.log(current/history)).mean()) if k else None,
            'temporal_relative_on_valid':float(relative.mean()) if k else None,
            'temporal_change_rate_m_per_s_on_valid':float(error.mean()/dt) if k else None,
            'capped_l1_m_full_reference':float((np.minimum(error,error_cap_m).sum()+(n-k)*error_cap_m)/n) if n else None,
            'agreement_5pct_full_reference':int((relative<.05).sum())/n if n else None})
    n=sum(r['reference_pixels'] for r in rows);k=sum(r['valid_pairs'] for r in rows)
    def pooled(key,count):
        denominator=sum(r[count] for r in rows if r[key] is not None)
        return float(sum(r[key]*r[count] for r in rows if r[key] is not None)/denominator) if denominator else None
    medians=[float(np.median(d[v])) if v.any() else None for d,v in zip(depths,valid)]
    jumps=[abs(np.log(b/a)) for a,b in zip(medians[:-1],medians[1:]) if a is not None and b is not None]
    all_valid=valid.all(axis=0)
    static_std=float(np.std(depths[:,all_valid].astype(np.float64),axis=0).mean()) if static_scene and all_valid.any() else None
    summary={'frames':len(depths),'pairs':len(rows),'reference_pixels':n,'valid_pairs':k,
        'reference_fraction_of_all_pairs':n/common_mask.size,'pair_coverage':k/n if n else None,
        'empty_reference_pairs':sum(r['reference_pixels']==0 for r in rows),
        **{key:pooled(key,'valid_pairs') for key in ('temporal_l1_m_on_valid','temporal_abs_log_on_valid','temporal_relative_on_valid','temporal_change_rate_m_per_s_on_valid')},
        **{key:pooled(key,'reference_pixels') for key in ('capped_l1_m_full_reference','agreement_5pct_full_reference')},
        'error_cap_m':error_cap_m,'frame_prediction_coverage':[float(v.mean()) for v in valid],
        'frame_median_m':medians,'median_abs_log_change':float(np.mean(jumps)) if jumps else None,
        'median_change_pairs':len(jumps),'static_pixel_std_m':static_std,
        'all_frames_valid_pixel_fraction':float(all_valid.mean()),
        'interpretation':'agreement proxy; camera/object Z changes are real signal; lower is not proof of accuracy',
        'aggregation':'pixel-pooled adjacent-pair metrics; fixed RGB/flow reference; invalid pairs pay capped error and fail agreement'}
    return summary,rows
