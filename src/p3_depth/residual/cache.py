"""One immutable shared prediction cache; GT never enters model inference."""
import json
from pathlib import Path
import time
import h5py
import numpy as np
import torch
import yaml
from p3_depth.data.nyuv2 import NYUv2Labeled, read_standard_split
from p3_depth.models.runtime import ImageModel
from .records import sha, write_json, snapshot, identity, append_json


def validate_split(root, study):
    split = json.loads((root / study['split']).read_text(encoding='utf-8'))
    fits, vals, tests = [split[k] for k in ('fit_ids', 'validation_ids', 'test_ids')]
    if split['index_base'] != 0 or [len(x) for x in (fits, vals, tests)] != [577, 218, 654]:
        raise ValueError('Expected 577/218/654 zero-based IDs')
    if any(len(set(x)) != len(x) for x in (fits, vals, tests)) or set(fits) & set(vals):
        raise ValueError('Duplicate/overlapping split')
    standard = read_standard_split(root / 'data/nyuv2/splits.mat')
    if set(fits + vals) != set(standard['train'] - 1) or set(tests) != set(standard['test'] - 1):
        raise ValueError('Standard split mismatch')
    for path, expected in [(root / study['data'], split['dataset_sha256']),
                           (root / 'data/nyuv2/splits.mat', split['split_sha256'])]:
        if sha(path) != expected:
            raise ValueError('Original dataset identity changed')
    with NYUv2Labeled(root / study['data']) as data:
        def scenes(ids):
            return {''.join(chr(int(c)) for c in data.file[data.file['scenes'][0, i]][()].ravel()) for i in ids}
        fs, vs = scenes(fits), scenes(vals)
        if fs & vs:
            raise ValueError('Fitting/validation scene overlap')
    return split, {'fitting_scenes': len(fs), 'validation_scenes': len(vs), 'scene_overlap': 0}


def prepare_cache(root, study, split_name='development', limit=None):
    split, split_check = validate_split(root, study)
    ids = sorted(split['fit_ids'] + split['validation_ids']) if split_name == 'development' else split['test_ids']
    if limit:
        ids = ids[:limit]
    label = split_name + (f'_smoke{limit}' if limit else '')
    output = root / study['output'] / 'cache' / label
    output.mkdir(parents=True, exist_ok=True)
    config = {'ids': ids, 'split': split_name, 'split_manifest_sha256': sha(root / study['split']),
              'dataset_sha256': split['dataset_sha256'], 'feature_hw': study['feature_hw'],
              'schema': 1, 'inference': 'official UniDepthV2 Small level=0 fp16, original RGB and horizontal flip',
              'internal_features': 'official infer depth_features: decoder initial latents; native grid, not GT-derived'}
    metadata_path, path = output / 'metadata.json', output / 'predictions.h5'
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if metadata['identity'] != identity(config):
            raise ValueError('Cache identity differs; use a different directory')
        if metadata['status'] == 'complete':
            if sha(path) != metadata['cache_sha256']:
                raise ValueError('Cache hash mismatch')
            print(f'PASS reused {label}: {len(ids)} verified records', flush=True)
            return
    settings = yaml.safe_load((root / 'configs/paths.local.yaml').read_text(encoding='utf-8'))
    adapter = ImageModel('unidepth', root, settings, 'cuda', precision='fp16', threads=4)
    adapter.model.requires_grad_(False)
    snapshot(root, output)
    metadata = {'status': 'running', 'identity': identity(config), 'config': config,
                'split_check': split_check, 'model': adapter.metadata,
                'timing': 'synchronized original/flip inference separately; excludes load and IO; three warmups',
                'gt_used_in_inference': False}
    write_json(metadata_path, metadata)
    with NYUv2Labeled(root / study['data']) as data, h5py.File(path, 'a') as dest:
        warm = torch.from_numpy(data.load(split['fit_ids'][0] + 1).rgb).permute(2, 0, 1)
        for _ in range(3):
            adapter.model.infer(warm)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for position, index in enumerate(ids):
            key = str(index)
            if key in dest and dest[key].attrs.get('complete', False):
                continue
            if key in dest:
                del dest[key]
            sample = data.load(index + 1)
            tensor = torch.from_numpy(sample.rgb).permute(2, 0, 1)
            torch.cuda.synchronize()
            start = time.perf_counter()
            original = adapter.model.infer(tensor)
            torch.cuda.synchronize()
            t_original = time.perf_counter() - start
            start = time.perf_counter()
            flipped = adapter.model.infer(tensor.flip(-1))
            torch.cuda.synchronize()
            t_flip = time.perf_counter() - start
            arrays = {'original': original['depth'][0, 0].float().cpu().numpy(),
                      'flip': flipped['depth'][0, 0].flip(-1).float().cpu().numpy(),
                      'confidence': original['confidence'][0, 0].float().cpu().numpy(),
                      'internal': original['depth_features'][0].float().cpu().numpy()}
            for name, value in arrays.items():
                if not np.isfinite(value).all() or (name != 'internal' and np.any(value <= 0)):
                    raise ValueError(f'Invalid {name}: {sample.sample_id}')
                if name != 'internal' and value.shape != (480, 640):
                    raise ValueError('Inference shape mismatch')
            group = dest.create_group(key)
            for name, value in arrays.items():
                # Preserve metric predictions in float32; FP16 internal features were produced by autocast.
                group.create_dataset(name, data=value.astype(np.float16 if name == 'internal' else np.float32), compression='lzf')
            group.attrs.update(scene=sample.scene, sample_id=sample.sample_id,
                               original_seconds=t_original, flip_seconds=t_flip, complete=True)
            dest.flush()
            append_json(output / 'records.jsonl', {'index': index, 'sample_id': sample.sample_id,
                        'original_s': t_original, 'flip_s': t_flip,
                        'internal_shape': list(arrays['internal'].shape)})
            if position % 25 == 0 or position + 1 == len(ids):
                print(f'CACHE {position+1}/{len(ids)} {sample.sample_id}; feature={arrays["internal"].shape}', flush=True)
        metadata.update(status='complete', images=len(ids), peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    metadata['cache_sha256'] = sha(path)
    metadata['bytes'] = path.stat().st_size
    write_json(metadata_path, metadata)
    print(f'PASS {label}: {len(ids)} frozen prediction pairs; {metadata["bytes"]/2**30:.2f} GiB', flush=True)
