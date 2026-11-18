"""Fitting-only edge feature calibration on RGB and frozen predictions, never GT."""
import json
import cv2
import h5py
import numpy as np
from p3_depth.data.nyuv2 import NYUv2Labeled
from .records import write_json, sha, snapshot


def derivatives(rgb, prediction):
    gray = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (5,5), 1.)
    rgb_magnitude = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    depth = cv2.GaussianBlur(np.log(np.maximum(prediction, .001)).astype(np.float32), (5,5), 1.)
    dx, dy = [cv2.Sobel(depth, cv2.CV_32F, x, y, ksize=3) for x, y in [(1,0),(0,1)]]
    scaled = [np.rint(value*512) for value in (dx,dy)]
    if any(np.max(np.abs(value)) > 32767 for value in scaled):
        raise ValueError('Canny derivative overflow; do not silently clip')
    return gray, rgb_magnitude, np.hypot(dx,dy), [value.astype(np.int16) for value in scaled]


def depth_canny(gradients, threshold):
    return cv2.Canny(gradients[0], gradients[1], .4*threshold*512, threshold*512, L2gradient=True) > 0


def prepare_edges(root, study):
    directory = root / study['output'] / 'edge_features'
    directory.mkdir(parents=True, exist_ok=True)
    cache_dir = root / study['output'] / 'cache/development'
    metadata = json.loads((cache_dir / 'metadata.json').read_text(encoding='utf-8'))
    if metadata['status'] != 'complete' or sha(cache_dir / 'predictions.h5') != metadata['cache_sha256']:
        raise ValueError('Unverified prediction cache')
    split = json.loads((root / study['split']).read_text(encoding='utf-8'))
    ids, fitting = metadata['config']['ids'], set(split['fit_ids'])
    configuration = {'cache_sha256': metadata['cache_sha256'], 'fitting_ids': sorted(fitting),
                     'feature_hw': study['feature_hw'], 'rgb_target_density': .05, 'depth_target_density': .02,
                     'domain': 'fixed crop [45:471,41:601]; no GT values or masks used',
                     'smoothing': 'Gaussian 5x5 sigma1 on uint8 gray / float32 log(predicted meters)',
                     'normalization': 'magnitude / fitting q95 RGB or q98 predicted depth, clipped [0,5]',
                     'resize': 'full-resolution features INTER_AREA to feature resolution; binary edges become occupancy fractions',
                     'canny': 'low=.4*high; L2; depth uses global derivative scale512; no per-image rescaling'}
    final = directory / 'metadata.json'
    if final.exists():
        stored = json.loads(final.read_text(encoding='utf-8'))
        if stored['config'] != configuration or sha(directory / 'features.h5') != stored['sha256']:
            raise ValueError('Edge cache changed')
        print('PASS reused calibrated edge features', flush=True)
        return
    snapshot(root, directory)
    rgb_grid = np.geomspace(8., 512., 25)
    depth_grid = np.geomspace(.005, 2., 25)
    rgb_counts, depth_counts = np.zeros(25, np.int64), np.zeros(25, np.int64)
    rgb_hist, depth_hist = np.zeros(8192, np.int64), np.zeros(32768, np.int64)
    domain = np.zeros((480,640), bool)
    domain[45:471,41:601] = True
    total = 0
    with h5py.File(cache_dir / 'predictions.h5', 'r') as cache, NYUv2Labeled(root / study['data']) as source:
        for number, index in enumerate(sorted(fitting), 1):
            sample = source.load(index+1)
            gray, rgb_mag, depth_mag, grads = derivatives(sample.rgb, cache[str(index)]['original'][:])
            for values, factor, hist in [(rgb_mag, 4., rgb_hist), (depth_mag, 4096., depth_hist)]:
                bins = np.floor(values[domain]*factor).astype(np.int64)
                if bins.max() >= len(hist):
                    raise ValueError('Gradient calibration histogram overflow')
                hist += np.bincount(bins, minlength=len(hist))
            for j, high in enumerate(rgb_grid):
                rgb_counts[j] += np.count_nonzero((cv2.Canny(gray, .4*high, high, L2gradient=True)>0) & domain)
            for j, high in enumerate(depth_grid):
                depth_counts[j] += np.count_nonzero(depth_canny(grads, high) & domain)
            total += int(domain.sum())
            if number % 50 == 0 or number == len(fitting):
                print(f'EDGE CALIBRATE {number}/{len(fitting)} fitting only', flush=True)
        calibration = {'rgb_high': float(rgb_grid[np.argmin(np.abs(rgb_counts/total-.05))]),
                       'depth_high': float(depth_grid[np.argmin(np.abs(depth_counts/total-.02))]),
                       'rgb_magnitude_normalizer': float(np.searchsorted(np.cumsum(rgb_hist), .95*total)/4),
                       'depth_magnitude_normalizer': float(np.searchsorted(np.cumsum(depth_hist), .98*total)/4096),
                       'rgb_grid': rgb_grid.tolist(), 'depth_grid': depth_grid.tolist(),
                       'rgb_densities': (rgb_counts/total).tolist(), 'depth_densities': (depth_counts/total).tolist()}
        write_json(directory / 'calibration.json', {'config': configuration, 'calibration': calibration})
        height, width = study['feature_hw']
        with h5py.File(directory / 'features.h5', 'w') as dest:
            dest.create_dataset('ids', data=np.array(ids))
            arrays = {name: dest.create_dataset(name, (len(ids),height,width), dtype=np.float16, compression='lzf')
                      for name in ('rgb_sobel','rgb_canny','depth_sobel','depth_canny')}
            for i, index in enumerate(ids):
                sample = source.load(index+1)
                gray, rm, dm, grads = derivatives(sample.rgb, cache[str(index)]['original'][:])
                values = {'rgb_sobel': np.clip(rm/max(calibration['rgb_magnitude_normalizer'],1e-8),0,5),
                          'depth_sobel': np.clip(dm/max(calibration['depth_magnitude_normalizer'],1e-8),0,5),
                          'rgb_canny': (cv2.Canny(gray,.4*calibration['rgb_high'],calibration['rgb_high'],L2gradient=True)>0).astype(np.float32),
                          'depth_canny': depth_canny(grads,calibration['depth_high']).astype(np.float32)}
                for name, value in values.items():
                    arrays[name][i] = cv2.resize(value, (width,height), interpolation=cv2.INTER_AREA)
                if (i+1) % 100 == 0 or i+1 == len(ids):
                    print(f'EDGE SAVE {i+1}/{len(ids)}', flush=True)
    write_json(final, {'status':'complete','config':configuration,'calibration':calibration,'sha256':sha(directory/'features.h5')})
    print('PASS fitting-calibrated RGB/prediction edges; no GT features',flush=True)


def load_edge_features(root, study, pack):
    directory = root / study['output'] / 'edge_features'
    meta = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
    if meta['status'] != 'complete' or meta['config']['cache_sha256'] != pack['metadata']['cache_sha256'] or sha(directory/'features.h5') != meta['sha256']:
        raise ValueError('Edge features do not match this prediction cache')
    with h5py.File(directory/'features.h5','r') as source:
        if list(source['ids'][:]) != pack['ids']:
            raise ValueError('Edge image ordering differs')
        return {key:source[key][:].astype(np.float32) for key in ('rgb_sobel','rgb_canny','depth_sobel','depth_canny')}
