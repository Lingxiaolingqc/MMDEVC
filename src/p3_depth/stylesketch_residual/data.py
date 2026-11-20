"""Frozen, native UniDepth features; no learned preprocessing or GT in features."""
import hashlib
import json
import time
import h5py
import numpy as np
import torch
import yaml
from p3_depth.models.runtime import ImageModel
from p3_depth.data.nyuv2 import NYUv2Labeled
from p3_depth.residual.records import sha, write_json
from p3_depth.residual_inputs.data import observable_inputs
from p3_depth.evaluation.boundary import boundary_masks
from p3_depth.evaluation.rgb_depth_edges import near_edges

MODULES = {'early': 'ups.0', 'early_skip': 'process_features.1',
           'mid': 'ups.1', 'mid_skip': 'process_features.2',
           'late': 'ups.2', 'late_projected': 'depth_mlp.2'}


def state_digest(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class Capture:
    def __init__(self, model):
        self.values = {}
        layer = model.pixel_decoder.depth_layer
        self.handles = [layer.get_submodule(path).register_forward_hook(self.hook(name))
                        for name, path in MODULES.items()]

    def hook(self, name):
        def receive(module, args, value):
            value = value.permute(0, 3, 1, 2) if name == 'late_projected' else value
            self.values[name] = value.detach()
        return receive

    def close(self):
        for handle in self.handles:
            handle.remove()


def prepare(root, study):
    out = root/study['output']/'features'
    out.mkdir(parents=True, exist_ok=True)
    prior = root/study['parent_output']/'cache/development'
    meta = json.loads((prior/'metadata.json').read_text())
    split = json.loads((root/study['split']).read_text())
    assert sha(root/study['data']) == split['dataset_sha256']
    assert sha(prior/'predictions.h5') == meta['cache_sha256']
    assert meta['config']['split_manifest_sha256'] == sha(root/study['split'])
    ids = meta['config']['ids']
    assert ids == sorted(split['fit_ids'] + split['validation_ids']) and len(ids) == 795
    assert not set(ids) & set(split['test_ids'])
    if (out/'metadata.json').exists():
        saved = json.loads((out/'metadata.json').read_text())
        if saved['status'] == 'complete':
            assert saved['parent_cache_sha256'] == meta['cache_sha256']
            assert all(sha(out/name) == digest for name, digest in saved['sha256'].items())
            print('PASS reused complete feature cache', flush=True)
            return saved
        raise FileExistsError('Incomplete feature cache retained; inspect failure before restarting')
    settings = yaml.safe_load((root/'configs/paths.local.yaml').read_text())
    adapter = ImageModel('unidepth', root, settings, 'cuda', precision='fp16', threads=4)
    adapter.model.eval().requires_grad_(False)
    assert not any(p.requires_grad for p in adapter.model.parameters())
    assert not any(m.training for m in adapter.model.modules())
    before = state_digest(adapter.model)
    capture = Capture(adapter.model)
    from unidepth.models.unidepthv2.unidepthv2 import get_paddings
    pads, shape = get_paddings((480, 640), adapter.model.shape_constraints['ratio_bounds'])
    assert not any(pads)
    write_json(out/'metadata.json', {'status': 'running', 'ids': ids, 'modules': MODULES})
    arrays, max_difference = {}, 0.
    shapes = {'init': (256, 31, 41), 'early': (256, 62, 82), 'early_skip': (256, 62, 82),
              'mid': (128, 124, 164), 'mid_skip': (128, 124, 164),
              'late': (64, 248, 328), 'late_projected': (64, 248, 328)}
    tick = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    pair_differences = {str(pair): [] for pair in study['candidate']['pairs']}
    with NYUv2Labeled(root/study['data']) as source, h5py.File(prior/'predictions.h5', 'r') as old:
        for n, index in enumerate(ids):
            rgb = source.file['images'][index].transpose(2, 1, 0).copy()
            with torch.inference_mode():
                result = adapter.model.infer(torch.from_numpy(rgb).permute(2, 0, 1))
            depth = result['depth'][0, 0].float().cpu().numpy()
            difference = float(np.max(np.abs(depth-old[str(index)]['original'][:])))
            assert difference <= 1e-5, (index, difference)
            max_difference = max(max_difference, difference)
            values = {'init': result['depth_features'][0].cpu().numpy(),
                      **{k: v[0].cpu().numpy() for k, v in capture.values.items()}}
            assert np.array_equal(values['init'].astype(np.float16), old[str(index)]['internal'][:])
            for pair in study['candidate']['pairs']:
                a, b = [values[name] for name in pair]
                delta = float(np.max(np.abs(a.astype(np.float32)-b.astype(np.float32))))
                assert delta > 0, 'Duplicated features cannot form a pair'
                pair_differences[str(pair)].append(delta)
            for name, value in values.items():
                assert value.shape == shapes[name] and np.isfinite(value).all(), (name, value.shape)
                if name not in arrays:
                    arrays[name] = np.lib.format.open_memmap(out/(name+'.npy'), mode='w+', dtype=np.float16, shape=(len(ids), *value.shape))
                arrays[name][n] = value
            if n == 0 or (n+1) % 50 == 0 or n+1 == len(ids):
                print(f'FEATURES {n+1}/{len(ids)}; native shapes {shapes}', flush=True)
        for value in arrays.values():
            value.flush()
    capture.close()
    after = state_digest(adapter.model)
    assert before == after and all(p.grad is None for p in adapter.model.parameters())
    record = {'status': 'complete', 'ids': ids, 'modules': MODULES,
              'init_source': 'infer.depth_features = depth_layer.to_latents(conditioned_features[0])',
              'late_projected_layout': 'depth_mlp.2 output NHWC -> NCHW; real LayerNorm+Linear output',
              'pair_max_abs_difference_minimum': {k: min(v) for k, v in pair_differences.items()},
              'shapes': {k: list(v.shape) for k, v in arrays.items()}, 'model': adapter.metadata,
              'parent_cache_sha256': meta['cache_sha256'], 'split_sha256': sha(root/study['split']),
              'foundation_state_before': before, 'foundation_state_after': after,
              'foundation_frozen': True, 'gt_used_in_features': False, 'padding': list(pads),
              'original_depth_max_abs_difference': max_difference, 'init_fp16_bitwise_equal': True,
              'elapsed_seconds': time.perf_counter()-tick,
              'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
              'sha256': {k+'.npy': sha(out/(k+'.npy')) for k in arrays}}
    write_json(out/'metadata.json', record)
    return record


def load_pack(root, study):
    prior = root/study['parent_output']/'cache/development'
    meta = json.loads((prior/'metadata.json').read_text())
    split = json.loads((root/study['split']).read_text())
    directory = root/study['output']/'features'
    fm = json.loads((directory/'metadata.json').read_text())
    assert fm['status'] == 'complete' and fm['parent_cache_sha256'] == meta['cache_sha256']
    assert sha(prior/'predictions.h5') == meta['cache_sha256']
    assert sha(root/study['data']) == split['dataset_sha256']
    assert fm['split_sha256'] == sha(root/study['split'])
    ids = fm['ids']
    assert ids == sorted(split['fit_ids'] + split['validation_ids'])
    pack = {key: np.empty((len(ids), 480, 640), np.float32) for key in ('original', 'truth')}
    pack.update(mask=np.empty((len(ids), 480, 640), bool), band=np.empty((len(ids), 480, 640), bool))
    features = np.empty((len(ids), 4, *study['feature_hw']), np.float32)
    scenes = []
    with NYUv2Labeled(root/study['data']) as source, h5py.File(prior/'predictions.h5', 'r') as old:
        for n, index in enumerate(ids):
            sample = source.load(index+1)
            group = old[str(index)]
            assert group.attrs['complete'] and group.attrs['scene'] == sample.scene
            base = group['original'][:]
            pack['original'][n], pack['truth'][n], pack['mask'][n] = base, sample.depth_m, sample.valid_gt
            region, _, edges, _ = boundary_masks(sample.depth_m, sample.depth_m, sample.valid_gt, sample.valid_gt)
            pack['band'][n] = near_edges(edges, 2) & region
            features[n] = observable_inputs(sample.rgb, base, None, {'drop_rgb': False, 'extra_inputs': []}, study['feature_hw'], {})
            scenes.append(sample.scene)
            if (n+1) % 200 == 0 or n+1 == len(ids):
                print(f'LOAD {n+1}/{len(ids)}', flush=True)
    positions = {index: n for n, index in enumerate(ids)}
    fits = [positions[i] for i in split['fit_ids']]
    vals = [positions[i] for i in split['validation_ids']]
    assert len(fits) == 577 and len(vals) == 218
    assert not {scenes[i] for i in fits} & {scenes[i] for i in vals}
    internal = {}
    for name in study['candidate']['channels']:
        assert sha(directory/(name+'.npy')) == fm['sha256'][name+'.npy']
        internal[name] = np.load(directory/(name+'.npy'), mmap_mode='r')
    pack.update(ids=ids, scenes=scenes, fitting=fits, validation=vals, metadata=meta, feature_metadata=fm)
    return pack, features, internal
