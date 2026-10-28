"""Explicit project depth-discontinuity metric with a fixed GT evaluation domain."""

import numpy as np
from scipy.ndimage import binary_erosion,distance_transform_edt
from p3_depth.temporal.fusion import log_edge_strength


def evaluate_boundary(prediction,gt,region,valid,relative_jump=.1,tolerance_px=2.,distance_cap_px=10.):
    if prediction.ndim!=2 or any(x.shape!=prediction.shape for x in (gt,region,valid)) or region.dtype!=np.bool_ or valid.dtype!=np.bool_:
        raise ValueError('Expected same-shape HW arrays and boolean masks')
    if relative_jump<=0 or tolerance_px<0 or distance_cap_px<=tolerance_px:
        raise ValueError('Invalid boundary thresholds')
    if np.any(region & (~np.isfinite(gt)|(gt<=0))) or np.any(valid & (~np.isfinite(prediction)|(prediction<=0))):
        raise ValueError('Masks must describe positive finite depths')
    domain=binary_erosion(region,structure=np.ones((3,3)),border_value=0)
    prediction_domain=domain & binary_erosion(valid,structure=np.ones((3,3)),border_value=0)
    threshold=np.log1p(relative_jump)
    gt_edges=(log_edge_strength(gt,region)>threshold)&domain
    pred_edges=(log_edge_strength(prediction,valid)>threshold)&prediction_domain
    ng,npred=int(gt_edges.sum()),int(pred_edges.sum())
    if npred:
        gt_dist=distance_transform_edt(~pred_edges)[gt_edges]
    else:
        gt_dist=np.full(ng,distance_cap_px)
    if ng:
        pred_dist=distance_transform_edt(~gt_edges)[pred_edges]
    else:
        pred_dist=np.full(npred,distance_cap_px)
    precision=float((pred_dist<=tolerance_px).mean()) if npred else (0. if ng else None)
    recall=float((gt_dist<=tolerance_px).mean()) if ng else None
    f1=2*precision*recall/(precision+recall) if precision is not None and recall is not None and precision+recall else (0. if ng else None)
    return {'gt_edge_pixels':ng,'prediction_edge_pixels':npred,
        'matched_gt_edge_pixels':int((gt_dist<=tolerance_px).sum()),
        'matched_prediction_edge_pixels':int((pred_dist<=tolerance_px).sum()),'domain_pixels':int(domain.sum()),
        'prediction_domain_coverage':float(prediction_domain.sum()/domain.sum()) if domain.any() else None,
        'precision':precision,'recall':recall,'f1':f1,
        'gt_to_prediction_capped_distance_px':float(np.minimum(gt_dist,distance_cap_px).mean()) if ng else None,
        'relative_jump':relative_jump,'tolerance_px':tolerance_px,'distance_cap_px':distance_cap_px}
