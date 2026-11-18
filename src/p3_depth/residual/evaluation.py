"""Course metrics plus fixed boundary diagnostics; no GT-based prediction alignment."""
import csv
import numpy as np
from p3_depth.evaluation.course import score_course_nyu
from p3_depth.evaluation.boundary import evaluate_boundary
from .records import write_json


def score_image(prediction, pack, i):
    truth, mask, band = pack['truth'][i], pack['mask'][i], pack['band'][i]
    result = {'sample_id': f'nyuv2_{pack["ids"][i]+1:04d}', 'source_index': pack['ids'][i], 'scene': pack['scenes'][i],
              **score_course_nyu(prediction, truth)}
    clipped = np.clip(prediction, .001, 10).astype(np.float32)
    boundary = evaluate_boundary(clipped, truth, mask, np.isfinite(clipped) & (clipped > 0))
    result.update({'boundary_' + key: value for key, value in boundary.items()})
    for name, selected in [('boundary_band', mask & band), ('nonboundary', mask & ~band)]:
        result[name + '_pixels'] = int(selected.sum())
        result[name + '_abs_rel'] = float(np.mean(np.abs(clipped[selected].astype(np.float64) - truth[selected])/truth[selected])) if selected.any() else None
    return result


def summarize(rows):
    result = {'images': len(rows), 'gt_pixels': sum(x['gt_pixels'] for x in rows)}
    for key in ['abs_rel', 'rmse_m', 'delta1', 'boundary_precision', 'boundary_recall', 'boundary_f1',
                'boundary_gt_to_prediction_capped_distance_px', 'boundary_band_abs_rel', 'nonboundary_abs_rel']:
        values = [x[key] for x in rows if x[key] is not None]
        result[key] = float(np.mean(values)) if values else None
        result[key + '_contributing_images'] = len(values)
    for key in ['clipped_low_pixels', 'clipped_high_pixels', 'boundary_gt_edge_pixels', 'boundary_prediction_edge_pixels',
                'boundary_matched_gt_edge_pixels', 'boundary_matched_prediction_edge_pixels']:
        result[key] = sum(x[key] for x in rows)
    gt, pred = result['boundary_gt_edge_pixels'], result['boundary_prediction_edge_pixels']
    p = result['boundary_matched_prediction_edge_pixels']/pred if pred else 0.
    r = result['boundary_matched_gt_edge_pixels']/gt if gt else 0.
    result.update(boundary_pooled_precision=p, boundary_pooled_recall=r, boundary_pooled_f1=2*p*r/(p+r) if p+r else 0.)
    result['aggregation'] = 'unweighted image means; pooled boundary scores explicitly named; no images omitted from depth metrics'
    return result


def save_evaluation(output, rows):
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'per_image.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = summarize(rows)
    write_json(output / 'metrics.json', result)
    return result
