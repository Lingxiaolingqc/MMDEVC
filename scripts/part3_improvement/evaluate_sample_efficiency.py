"""Bridge cached/live inference, then freeze and evaluate all declared test checkpoints.

Every GPU phase is guarded against concurrent study training. Test inference cannot
start until all 72 development stages, validation analysis, and the bridge pass.
"""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from p3_depth.sample_efficiency.protocol import load, outdir, jobs, name, check_freeze
from p3_depth.residual.records import sha, write_json, append_json, now, identity
from p3_depth.residual.locking import study_lock


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def require_completed():
    from p3_depth.sample_efficiency.training import verify_result
    cfg = load(ROOT); check_freeze(ROOT)
    return [verify_result(ROOT, job) for job in jobs(cfg)]


class OnlinePack(dict):
    """Only one sample resident; model receives RGB and no ground-truth channels."""
    def __init__(self, ids):
        from p3_depth.data.nyuv2 import NYUv2Labeled
        self.cfg = load(ROOT)
        self.source = NYUv2Labeled(ROOT / self.cfg['data'])
        super().__init__(ids=ids, scenes=[''] * len(ids))
        self.current = None

    def select(self, i):
        if self.current != i:
            from p3_depth.evaluation.boundary import boundary_masks
            from p3_depth.evaluation.rgb_depth_edges import near_edges
            from p3_depth.residual_inputs.data import observable_inputs
            self.sample = self.source.load(self['ids'][i] + 1)
            self['scenes'][i] = self.sample.scene
            depth, valid = self.sample.depth_m, self.sample.valid_gt
            region, _, edges, _ = boundary_masks(depth, depth, valid, valid)
            self['truth'] = {i: depth}; self['mask'] = {i: valid}
            self['band'] = {i: near_edges(edges, 2) & region}
            # The fourth channel is a placeholder. J/S replaces it with live log D.
            self.inputs = observable_inputs(self.sample.rgb, np.ones((480, 640), np.float32), None,
                                             self.cfg['refiner'], self.cfg['feature_hw'], {})
            self.current = i
        return self.sample

    def rgb(self, positions):
        import torch
        assert len(positions) == 1
        sample = self.select(positions[0])
        return torch.from_numpy(sample.rgb.transpose(2, 0, 1)[None].copy()).cuda()

    def tensor(self, key, positions):
        import torch
        assert key == 'inputs' and len(positions) == 1
        self.select(positions[0])
        return torch.from_numpy(self.inputs[None]).cuda()

    def close(self):
        self.source.file.close()


class LiveResidual:
    """Faithful live version of R: same NumPy input construction and FP16 cache precision."""
    def __init__(self, cfg, seed, state):
        from p3_depth.sample_efficiency.model import Pipeline
        self.ref = Pipeline(ROOT, cfg, 'R', seed); self.ref.restore(state); self.ref.refiner.eval()
        self.base = Pipeline(ROOT, cfg, 'B0', seed)
        self.cfg = cfg; self.values = {}; self.handles = []
        for key, index in [('mid', -2), ('late', -1)]:
            def capture(module, args, output, key=key): self.values[key] = output
            self.handles.append(self.base.foundation.pixel_decoder.depth_layer.ups[index].register_forward_hook(capture))

    def predict(self, pack, positions):
        import torch
        from p3_depth.residual_inputs.data import observable_inputs
        from p3_depth.residual.model import corrected_depth
        result = self.base.infer(pack.rgb(positions))
        depth = result['depth'].float()
        if hasattr(pack, 'sample'):
            rgb = pack.sample.rgb
        else:
            rgb = pack.file['images'][pack['ids'][positions[0]]].transpose(2, 1, 0).copy()
        inputs = observable_inputs(rgb, depth[0, 0].cpu().numpy(), None,
                                   self.cfg['refiner'], self.cfg['feature_hw'], {})
        latent = {'init': result['depth_features'].half().float(),
                  **{k: v.half().float() for k, v in self.values.items()}}
        self.values.clear()
        with torch.autocast('cuda', dtype=torch.float16):
            residual = self.ref.refiner(torch.from_numpy(inputs[None]).cuda(), latent)
        return corrected_depth(depth, residual, 'additive')

    def buffers(self):
        return {**self.ref.buffers(), **self.base.buffers()}

    def close(self):
        for handle in self.handles: handle.remove()
        self.values.clear(); self.ref.close(); self.base.close()


def bridge():
    """Validation-only numerical equivalence; thresholds fixed before any test inference."""
    import torch
    from p3_depth.sample_efficiency.training import setup
    from p3_depth.sample_efficiency.model import Pipeline
    from p3_depth.sample_efficiency.data import Pack
    cfg = load(ROOT); require_completed(); study = outdir(ROOT)
    setup(cfg['seeds'][0]); cache = Pack(ROOT)
    ids = read(study / 'subsets.json')['validation_ids']; online = OnlinePack(ids)
    job = dict(method='R', count=100, seed=cfg['seeds'][0])
    state = torch.load(study / 'runs' / name(job) / 'best.pt', map_location='cpu', weights_only=False)['pipeline']
    cached = Pipeline(ROOT, cfg, 'R', job['seed']); cached.restore(state); cached.refiner.eval()
    live = LiveResidual(cfg, job['seed'], state); maximum = 0.
    try:
        with torch.no_grad():
            for j, i in enumerate(cache.positions(ids)):
                online.select(j)
                assert np.array_equal(cache['truth'][i], online['truth'][j])
                assert np.array_equal(cache['mask'][i], online['mask'][j])
                assert np.array_equal(cache['band'][i], online['band'][j])
                assert np.array_equal(cache['inputs'][i][:3], online.inputs[:3])
                a = cached.predict(cache, [i]); b = live.predict(online, [j])
                difference = float((a-b).abs().max()); maximum = max(maximum, difference)
                assert difference <= 1e-5, ('Cached/live R mismatch', ids[j], difference)
        record = {'status': 'pass', 'at': now(), 'validation_images': len(ids), 'maximum_prediction_difference_m': maximum,
                  'absolute_prediction_tolerance_m': 1e-5, 'input_truth_mask_band_parity': 'exact',
                  'job': job, 'checkpoint_sha256': sha(study / 'runs' / name(job) / 'best.pt'),
                  'freeze_sha256': sha(study / 'frozen.json'), 'evaluation_script_sha256': sha(Path(__file__))}
        write_json(study / 'live_bridge.json', record)
    finally:
        cached.close(); live.close(); cache.close(); online.close()


def freeze_test():
    cfg = load(ROOT); results = require_completed(); study = outdir(ROOT)
    bridge_record = read(study / 'live_bridge.json')
    assert bridge_record['status'] == 'pass' and bridge_record['evaluation_script_sha256'] == sha(Path(__file__))
    analysis = read(study / 'analysis' / 'summary.json')
    assert analysis['completed'] == 72 and analysis['freeze_sha256'] == sha(study / 'frozen.json')
    # Human/assistant review is an explicit deliverable, not inferred from image creation.
    review = read(study / 'analysis' / 'visual_review.json')
    assert review['status'] == 'reviewed' and review['files_reviewed']
    split = read(study / 'subsets.json'); entries = [{'job': 'B0', 'checkpoints': {}}]
    for result in results:
        entries.append({'job': result['job'], 'checkpoints': result['checkpoint_sha256']})
    record = {'status': 'frozen_before_this_study_test_inference', 'at': now(), 'entries': entries,
              'test_ids': split['test_ids'], 'source_freeze_sha256': sha(study / 'frozen.json'),
              'live_bridge_sha256': sha(study / 'live_bridge.json'), 'analysis_sha256': sha(study / 'analysis' / 'summary.json'),
              'visual_review_sha256': sha(study / 'analysis' / 'visual_review.json'),
              'evaluation_script_sha256': sha(Path(__file__)),
              'test_history': 'Official test was inspected in earlier course/model studies. This is not a previously unseen blind test.',
              'selection': 'All 72 predefined jobs, both validation-best and stopping-final; no test selection or all795 refit',
              'fixed_test_preview_ids': [split['test_ids'][i]+1 for i in [0, len(split['test_ids'])//2, -1]]}
    target = study / 'test_frozen.json'
    if target.exists():
        old = read(target); assert {k:v for k,v in old.items() if k != 'at'} == {k:v for k,v in record.items() if k != 'at'}
    else:
        assert not (study / 'final_test').exists()
        write_json(target, record)


def checked_test():
    study = outdir(ROOT); check_freeze(ROOT); record = read(study / 'test_frozen.json')
    assert record['evaluation_script_sha256'] == sha(Path(__file__))
    assert record['source_freeze_sha256'] == sha(study / 'frozen.json')
    assert record['live_bridge_sha256'] == sha(study / 'live_bridge.json')
    assert len(record['test_ids']) == len(set(record['test_ids'])) == 654
    return record


def evaluate_job(job_name):
    import torch
    from p3_depth.sample_efficiency.training import setup
    from p3_depth.sample_efficiency.model import Pipeline
    from p3_depth.residual.evaluation import score_image, save_evaluation
    record = checked_test(); cfg = load(ROOT); study = outdir(ROOT)
    entry = next(e for e in record['entries'] if ('B0' if e['job'] == 'B0' else name(e['job'])) == job_name)
    seed = cfg['seeds'][0] if job_name == 'B0' else entry['job']['seed']; setup(seed)
    for checkpoint in (['original'] if job_name == 'B0' else ['best', 'final']):
        folder = study / 'final_test' / job_name / checkpoint; folder.mkdir(parents=True, exist_ok=True)
        expected = {'test_freeze_sha256': sha(study / 'test_frozen.json'), 'job': entry['job'], 'checkpoint': checkpoint,
                    'checkpoint_sha256': None if job_name == 'B0' else entry['checkpoints'][checkpoint+'.pt']}
        if (folder / 'result.json').exists():
            old = read(folder / 'result.json'); assert old['identity'] == expected and old['status'] == 'complete'
            for file, digest in old['files'].items(): assert sha(folder/file) == digest
            continue
        tick = time.perf_counter(); pack = OnlinePack(record['test_ids']); pipe = None
        try:
            if job_name == 'B0':
                pipe = Pipeline(ROOT, cfg, 'B0', seed)
            else:
                path = study / 'runs' / job_name / (checkpoint+'.pt')
                assert sha(path) == expected['checkpoint_sha256']
                saved = torch.load(path, map_location='cpu', weights_only=False)['pipeline']
                method = entry['job']['method']
                if method == 'R': pipe = LiveResidual(cfg, seed, saved)
                else:
                    pipe = Pipeline(ROOT, cfg, method, seed, saved if method == 'S' else None)
                    pipe.restore(saved)
                    if pipe.refiner: pipe.refiner.eval()
                del saved
            before = pipe.buffers(); rows = []; previews = {}; times = []
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                for i, index in enumerate(record['test_ids']):
                    pack.select(i); torch.cuda.synchronize(); started = time.perf_counter()
                    prediction = pipe.predict(pack, [i])[0, 0].cpu().numpy()
                    torch.cuda.synchronize(); times.append(time.perf_counter()-started)
                    assert np.isfinite(prediction).all()
                    rows.append(score_image(prediction, pack, i))
                    if index+1 in record['fixed_test_preview_ids']: previews[str(index+1)] = prediction
                    if (i+1) % 50 == 0:
                        write_json(folder / 'progress.json', {'images': i+1, 'total': 654, 'at': now()})
                        print(f'{job_name}/{checkpoint} {i+1}/654', flush=True)
            after = pipe.buffers(); assert before.keys() == after.keys() and all(torch.equal(before[k], after[k]) for k in before)
            metrics = save_evaluation(folder, rows); np.savez_compressed(folder/'fixed_predictions.npz', **previews)
            result = {'status': 'complete', 'identity': expected, 'metrics': metrics, 'at': now(),
                      'elapsed_seconds': time.perf_counter()-tick, 'mean_infer_seconds': float(np.mean(times)),
                      'timing_scope': 'RGB input already resident; model+input conversion+CPU depth return; excludes GT scoring/disk load; not isolated benchmark',
                      'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
                      'files': {f:sha(folder/f) for f in ['per_image.csv', 'metrics.json', 'fixed_predictions.npz']}}
            write_json(folder/'result.json', result)
        except BaseException as exc:
            append_json(study/'test_failures.jsonl', {'job':job_name, 'checkpoint':checkpoint, 'error':repr(exc)})
            raise
        finally:
            if pipe: pipe.close()
            pack.close(); del pipe, pack; gc.collect(); torch.cuda.empty_cache()


def queue():
    import shutil
    from run_sample_efficiency import memory
    record = checked_test(); study = outdir(ROOT); logs = study/'test_logs'; logs.mkdir(exist_ok=True)
    for entry in record['entries']:
        label = 'B0' if entry['job'] == 'B0' else name(entry['job'])
        limits = load(ROOT)['resource']
        while True:
            resources = memory(); free_disk = shutil.disk_usage(ROOT).free / 2**30
            if resources['commit_available_gib'] >= limits['minimum_available_commit_gib'] and free_disk >= limits['minimum_free_disk_gib']:
                break
            write_json(study/'test_progress.json', {'status':'waiting_resources', 'job':label,
                                                   'resources':resources, 'free_disk_gib':free_disk, 'at':now()})
            time.sleep(10)
        suffix = str(time.time_ns())
        with (logs/(label+'_'+suffix+'.stdout.log')).open('w', encoding='utf8') as stdout, \
             (logs/(label+'_'+suffix+'.stderr.log')).open('w', encoding='utf8') as stderr:
            process = subprocess.run([sys.executable, '-X', 'utf8', str(Path(__file__)), 'evaluate', '--job', label],
                                     cwd=ROOT, stdout=stdout, stderr=stderr)
        if process.returncode: raise RuntimeError('Test worker failed: '+label)
        write_json(study/'test_progress.json', {'last_completed':label, 'at':now()})
    write_json(study/'test_complete.json', {'at':now(), 'jobs':73, 'checkpoint_evaluations':145,
                                          'test_freeze_sha256':sha(study/'test_frozen.json')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['bridge', 'freeze', 'evaluate', 'queue'])
    parser.add_argument('--job')
    args = parser.parse_args(); study = outdir(ROOT)
    if args.phase == 'evaluate':
        with study_lock(study/'gpu_worker'): evaluate_job(args.job)
    else:
        with study_lock(study):
            if args.phase == 'bridge':
                with study_lock(study/'gpu_worker'): bridge()
            elif args.phase == 'freeze': freeze_test()
            else: queue()


if __name__ == '__main__': main()
