"""Shared development data and input features, with fitting-only calibration."""
import json
import cv2
import h5py
import numpy as np
from scipy.optimize import minimize_scalar
from p3_depth.data.nyuv2 import NYUv2Labeled, evaluation_mask
from p3_depth.evaluation.boundary import boundary_masks
from p3_depth.evaluation.rgb_depth_edges import near_edges
from .records import sha, write_json


def load_pack(root, study, subset='development'):
    directory = root / study['output'] / 'cache' / subset
    meta = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
    path = directory / 'predictions.h5'
    if meta['status'] != 'complete' or sha(path) != meta['cache_sha256']:
        raise ValueError('Incomplete/modified prediction cache')
    split = json.loads((root / study['split']).read_text(encoding='utf-8'))
    if sha(root/study['data']) != split['dataset_sha256']:
        raise ValueError('Original NYUv2 data changed since split/cache creation')
    if meta['config']['split_manifest_sha256'] != sha(root / study['split']):
        raise ValueError('Split changed after caching')
    ids = meta['config']['ids']
    count = len(ids)
    data = {key: np.empty((count, 480, 640), np.float32) for key in ('original', 'flip', 'truth')}
    data['mask'] = np.empty((count, 480, 640), bool)
    data['band'] = np.empty_like(data['mask'])
    height, width = study['feature_hw']
    data['rgb'] = np.empty((count, 3, height, width), np.float32)
    data['confidence'] = np.empty((count, height, width), np.float32)
    scenes = []
    with NYUv2Labeled(root / study['data']) as source, h5py.File(path, 'r') as cache:
        for i, index in enumerate(ids):
            group, sample = cache[str(index)], source.load(index + 1)
            if not group.attrs['complete'] or group.attrs['scene'] != sample.scene:
                raise ValueError('Cache record identity mismatch')
            for key in ('original', 'flip'):
                data[key][i] = group[key][:]
            data['truth'][i], data['mask'][i] = sample.depth_m, sample.valid_gt
            region, _, edges, _ = boundary_masks(sample.depth_m, sample.depth_m, sample.valid_gt, sample.valid_gt)
            data['band'][i] = near_edges(edges, 2) & region
            data['rgb'][i] = (cv2.resize(sample.rgb, (width, height), interpolation=cv2.INTER_AREA).transpose(2, 0, 1).astype(np.float32)/127.5 - 1)
            confidence = cv2.resize(group['confidence'][:], (width, height), interpolation=cv2.INTER_AREA)
            data['confidence'][i] = np.clip(np.log(np.maximum(confidence, 1e-6)), -5, 5)/5
            scenes.append(sample.scene)
            if (i+1) % 100 == 0 or i+1 == count:
                print(f'LOAD {i+1}/{count} study images', flush=True)
    positions = {index: i for i, index in enumerate(ids)}
    data.update(ids=ids, scenes=scenes, metadata=meta, cache_path=path,
                fitting=[positions[i] for i in split['fit_ids'] if i in positions],
                validation=[positions[i] for i in split['validation_ids'] if i in positions])
    return data


def fit_scale(base, truth, mask, positions):
    """Bounded scalar fit on every fitting GT pixel; equal weight per image."""
    if not positions:
        raise ValueError('No fitting data for calibration')
    def objective(log_scale):
        scale = float(np.exp(log_scale))
        scores = []
        for i in positions:
            target = truth[i][mask[i]].astype(np.float64)
            predicted = base[i][mask[i]].astype(np.float64)
            scores.append(np.mean(np.abs(np.clip(predicted * scale, .001, 10) - target)/target))
        return float(np.mean(scores))
    fitted = minimize_scalar(objective, method='bounded', bounds=(np.log(.25), np.log(4)), options={'xatol': 1e-5})
    if not fitted.success or abs(fitted.x) > np.log(4) - .001:
        raise ValueError('Scale fit failed/hit declared search bound')
    return {'scale': float(np.exp(fitted.x)), 'fitting_abs_rel': float(fitted.fun),
            'objective': 'full fitting-pixel course-clipped AbsRel, equal per-image weights',
            'scale_bounds': [.25, 4.], 'evaluations': int(fitted.nfev)}


def calibration(root, study, pack):
    path = root / study['output'] / 'calibration.json'
    cache_hash = pack['metadata']['cache_sha256']
    if path.exists():
        result = json.loads(path.read_text(encoding='utf-8'))
        if result['cache_sha256'] != cache_hash:
            raise ValueError('Calibration cache mismatch')
        return result
    result = {'cache_sha256': cache_hash, 'fitting_ids': [pack['ids'][i] for i in pack['fitting']]}
    for name, base in [('scale', pack['original']), ('flip_scale', .75 * pack['original'] + .25 * pack['flip'])]:
        result[name] = fit_scale(base, pack['truth'], pack['mask'], pack['fitting'])
        print(f'CALIBRATION {name}: {result[name]}', flush=True)
    write_json(path, result)
    return result


def base_for(pack, name, calibration):
    if name in ('original', 'scale'):
        base = pack['original']
    elif name in ('flip', 'flip_scale'):
        base = .75 * pack['original'] + .25 * pack['flip']
    else:
        raise ValueError('Unknown base')
    return base * calibration[name]['scale'] if name in ('scale', 'flip_scale') else base


def features_for(pack, config, base, feature_hw, edge_features=None):
    height, width = feature_hw
    def resized(values):
        return np.stack([cv2.resize(x, (width, height), interpolation=cv2.INTER_AREA) for x in values])
    def log_map(values):
        return (np.log(np.maximum(resized(values), .001))/2)[:, None]
    name = config['inputs']
    parts = [log_map(base)] if name == 'depth_only' else [pack['rgb'], log_map(base)]
    internal = None
    if name == 'flip':
        parts += [log_map(pack['flip'])]
    elif name == 'confidence':
        parts += [pack['confidence'][:, None]]
    elif name == 'coordinates':
        yy, xx = np.meshgrid(np.linspace(-1, 1, height), np.linspace(-1, 1, width), indexing='ij')
        parts += [np.broadcast_to(np.stack([xx, yy]), (len(base), 2, height, width))]
    elif name == 'flip_disagreement':
        q = log_map(pack['flip']) - log_map(pack['original'])
        parts += [q, np.abs(q)]
    elif name == 'legacy':
        yy, xx = np.meshgrid(np.linspace(-1, 1, height), np.linspace(-1, 1, width), indexing='ij')
        parts = [pack['rgb'], log_map(pack['original']), log_map(pack['flip']), log_map(base),
                 pack['confidence'][:, None], np.broadcast_to(np.stack([xx, yy]), (len(base), 2, height, width))]
    elif name == 'frozen_features':
        with h5py.File(pack['cache_path'], 'r') as source:
            internal = np.stack([source[str(i)]['internal'][:] for i in pack['ids']])
    elif name in ('rgb_sobel', 'rgb_canny', 'depth_sobel', 'depth_canny', 'joint_edges'):
        if edge_features is None:
            raise ValueError('Run fitting-only edge feature preparation before edge training')
        keys = ['rgb_sobel', 'depth_sobel'] if name == 'joint_edges' else [name]
        parts += [edge_features[key][:, None] for key in keys]
    elif name not in ('rgb_depth', 'depth_only'):
        raise ValueError('Unknown inputs')
    values = np.concatenate(parts, 1).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite input features')
    return values, internal
