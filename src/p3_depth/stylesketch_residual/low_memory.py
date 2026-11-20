"""Numerically unchanged, bounded-resident IO recovery for the frozen SS1 study.

This execution supplement changes storage/access only. Architecture, batches,
loss, optimizer, RNG and checkpoint selection remain the frozen original code.
"""
import json
import h5py
import numpy as np
import torch
from .common import check_freeze
from p3_depth.data.nyuv2 import NYUv2Labeled
from p3_depth.evaluation.boundary import boundary_masks
from p3_depth.evaluation.rgb_depth_edges import near_edges
from p3_depth.residual_inputs.data import observable_inputs
from p3_depth.residual.records import sha, write_json, append_json, now


class DiskArray:
    """Detach a selected slice and immediately release its file mapping."""
    def __init__(self, path):
        self.path = path
        view = np.load(path, mmap_mode='r')
        self.shape, self.dtype = view.shape, view.dtype
        view._mmap.close()

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, item):
        view = np.load(self.path, mmap_mode='r')
        try:
            return np.array(view[item], copy=True, order='C')
        finally:
            view._mmap.close()


VERIFIED = set()


def verify(path, expected):
    key = (str(path), expected)
    if key not in VERIFIED:
        assert sha(path) == expected, path
        VERIFIED.add(key)


def prepare(root, study):
    check_freeze(root, study)
    out = root/study['output']; cache = out/'disk_inputs'; cache.mkdir(exist_ok=True)
    prior = root/study['parent_output']/'cache/development'
    original = json.loads((prior/'metadata.json').read_text())
    fm = json.loads((out/'features/metadata.json').read_text())
    split = json.loads((root/study['split']).read_text())
    verify(root/study['data'], split['dataset_sha256'])
    verify(prior/'predictions.h5', original['cache_sha256'])
    assert fm['ids'] == original['config']['ids'] == sorted(split['fit_ids']+split['validation_ids'])
    assert fm['split_sha256'] == sha(root/study['split'])
    if (cache/'metadata.json').exists():
        meta = json.loads((cache/'metadata.json').read_text())
        assert meta['status'] == 'complete' and meta['parent_cache_sha256'] == original['cache_sha256']
        for name, digest in meta['sha256'].items():
            verify(cache/name, digest)
        return meta
    ids = fm['ids']; arrays = {}; scenes = []
    with NYUv2Labeled(root/study['data']) as source, h5py.File(prior/'predictions.h5', 'r') as old:
        for n, index in enumerate(ids):
            sample = source.load(index+1); group = old[str(index)]
            assert group.attrs['complete'] and group.attrs['scene'] == sample.scene
            base = group['original'][:]
            region, _, edges, _ = boundary_masks(sample.depth_m, sample.depth_m, sample.valid_gt, sample.valid_gt)
            values = {'original': base, 'truth': sample.depth_m, 'mask': sample.valid_gt,
                      'band': near_edges(edges, 2) & region,
                      'inputs': observable_inputs(sample.rgb, base, None, {'drop_rgb': False, 'extra_inputs': []}, study['feature_hw'], {})}
            for name, value in values.items():
                dtype = np.bool_ if name in ('mask', 'band') else np.float32
                if name not in arrays:
                    arrays[name] = np.lib.format.open_memmap(cache/(name+'.npy'), mode='w+', dtype=dtype, shape=(len(ids), *value.shape))
                arrays[name][n] = value
                assert np.array_equal(arrays[name][n], value.astype(dtype)), (name, index)
            scenes.append(sample.scene)
            if (n+1) % 200 == 0 or n+1 == len(ids):
                print(f'DISK INPUTS {n+1}/{len(ids)}; exact per-image source parity', flush=True)
    for value in arrays.values():
        value.flush(); value._mmap.close()
    meta = {'status': 'complete', 'ids': ids, 'scenes': scenes, 'parent_cache_sha256': original['cache_sha256'],
            'split_sha256': sha(root/study['split']), 'all795_source_values_bitwise_equal_after_write': True,
            'sha256': {name+'.npy': sha(cache/(name+'.npy')) for name in arrays}}
    write_json(cache/'metadata.json', meta)
    return meta


def load_pack(root, study):
    meta = prepare(root, study)
    out = root/study['output']; cache = out/'disk_inputs'
    fm = json.loads((out/'features/metadata.json').read_text())
    prior = root/study['parent_output']/'cache/development'
    original = json.loads((prior/'metadata.json').read_text())
    split = json.loads((root/study['split']).read_text())
    ids, scenes = meta['ids'], meta['scenes']; positions = {index: n for n, index in enumerate(ids)}
    fits = [positions[i] for i in split['fit_ids']]; vals = [positions[i] for i in split['validation_ids']]
    assert len(fits) == 577 and len(vals) == 218
    assert not {scenes[i] for i in fits} & {scenes[i] for i in vals}
    pack = {name: DiskArray(cache/(name+'.npy')) for name in ('original','truth','mask','band')}
    pack.update(ids=ids, scenes=scenes, fitting=fits, validation=vals, metadata=original, feature_metadata=fm)
    internal = {}
    for name in study['candidate']['channels']:
        path = out/'features'/(name+'.npy')
        verify(path, fm['sha256'][name+'.npy'])
        internal[name] = DiskArray(path)
    return pack, DiskArray(cache/'inputs.npy'), internal


def recovery_check(root, study):
    from .model import StyleSketchResidual
    from .training import checked_evaluation
    out = root/study['output']; pack, features, internal = load_pack(root, study)
    # Fancy indexing, scalar indexing and duplicates preserve values/order exactly.
    parity = []
    for array in [*internal.values(), features, pack['original'], pack['truth'], pack['mask'], pack['band']]:
        view = np.load(array.path, mmap_mode='r')
        for selection in ([2,0,2,400], 13):
            assert np.array_equal(array[selection], view[selection])
        view._mmap.close(); parity.append(array.path.name)
    run = out/'runs/SS1_seed320112'
    saved = torch.load(run/'best.pt', map_location='cpu', weights_only=False)
    model = StyleSketchResidual(study['candidate']).cuda().eval(); model.load_state_dict(saved['model'])
    scores, _ = checked_evaluation(model, study['candidate'], pack, features, internal)
    assert abs(scores['abs_rel']-saved['selection_abs_rel']) < 1e-7
    resume = torch.load(run/'resume.pt', map_location='cpu', weights_only=False)
    assert resume['epoch'] == json.loads((run/'history.json').read_text())[-1]['epoch']
    record = {'at': now(), 'reason': 'System RAM pressure, not CUDA OOM or changed candidate quality.',
              'observed_free_physical_memory_mib': 121.5859375,
              'resumed_from_epoch': resume['epoch'], 'resumed_updates': resume['updates'],
              'resume_sha256': sha(run/'resume.pt'), 'best_checkpoint_sha256': sha(run/'best.pt'),
              'stored_validation_abs_rel': saved['selection_abs_rel'], 'disk_reader_validation_abs_rel': scores['abs_rel'],
              'all795_source_values_bitwise_equal_after_write': True, 'indexing_parity': parity,
              'architecture_protocol_optimizer_batch_order_and_rng_unchanged': True,
              'partial_incomplete_epoch_discarded': True,
              'design_freeze_unchanged': sha(out/'design_frozen.json'),
              'runtime_source_sha256': {name: sha(root/name) for name in
                  ('src/p3_depth/stylesketch_residual/low_memory.py','scripts/part3_improvement/run_stylesketch_low_memory.py')}}
    path = out/'execution_supplement.json'
    assert not path.exists(); write_json(path, record)
    append_json(out/'recovery_events.jsonl', record)
    print('PASS identical saved-checkpoint validation through disk-backed reader', flush=True)
