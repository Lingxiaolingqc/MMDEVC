"""Cache fixed-resolution UniDepth predictions for the course development split."""
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from depth_project.data import load_nyu_split
from depth_project.models import load_image_adapter
from depth_project.records import run_provenance, sha256_file


def scene_name(dataset, index):
    return ''.join(chr(int(value)) for value in dataset[dataset['scenes'][0, index]][:].reshape(-1))


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / 'runs/nyu-development'
    output.mkdir(parents=True, exist_ok=True)
    split = load_nyu_split(root / 'data/nyu/splits.mat')
    torch.set_num_threads(4)
    adapter = load_image_adapter('unidepth', root)
    identities = {
        'checkpoint_sha256': adapter.metadata['checkpoint_sha256'],
        'dataset_sha256': sha256_file(root / 'data/nyu/nyu_depth_v2_labeled.mat'),
        'split_sha256': sha256_file(root / 'data/nyu/splits.mat'),
        'preprocessing': adapter.metadata['preprocessing_id'],
        'variants': ['original', 'horizontal_flip'],
    }
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['identities'] != identities:
            raise ValueError('Development cache identity differs')
    else:
        manifest = {'identities': identities, 'provenance': run_provenance(root)}
    latencies = []
    with h5py.File(root / 'data/nyu/nyu_depth_v2_labeled.mat', 'r') as dataset:
        scenes = {int(index): scene_name(dataset, int(index)) for index in split['development']}
        unique = np.array(sorted(set(scenes.values())))
        rng = np.random.default_rng(320112)
        rng.shuffle(unique)
        validation_scenes = set(unique[:max(1, round(len(unique) * .25))])
        manifest['fit_ids'] = [int(index) for index in split['development'] if scenes[int(index)] not in validation_scenes]
        manifest['validation_ids'] = [int(index) for index in split['development'] if scenes[int(index)] in validation_scenes]
        manifest['scenes'] = scenes
        manifest['validation_scenes'] = sorted(validation_scenes)
        manifest['seed'] = 320112
        assert set(manifest['fit_ids']).isdisjoint(manifest['validation_ids'])
        assert set(manifest['fit_ids'] + manifest['validation_ids']).isdisjoint(split['test'].tolist())
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        warm = np.ascontiguousarray(dataset['images'][int(split['development'][0])].transpose(2, 1, 0))
        for _ in range(2):
            adapter.predict_rgb(warm)
        for number, index in enumerate(split['development']):
            index = int(index)
            path = output / f'prediction_{index:04d}.npz'
            if path.exists():
                with np.load(path, allow_pickle=False) as cached:
                    assert int(cached['image_index']) == index
                    assert str(cached['checkpoint_sha256']) == identities['checkpoint_sha256']
                    assert cached['depth_m'].shape == (480, 640)
                    assert np.isfinite(cached['depth_m']).all() and np.isfinite(cached['flip_m']).all()
                continue
            rgb = np.ascontiguousarray(dataset['images'][index].transpose(2, 1, 0))
            tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).cuda()
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                original = adapter.model.infer(tensor)
                depth = original['depth'][0, 0].float().cpu().numpy()
                confidence = original['confidence'][0, 0].float().cpu().numpy()
                flipped = adapter.model.infer(torch.flip(tensor, [-1]))
                flip = torch.flip(flipped['depth'], [-1])[0, 0].float().cpu().numpy()
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - start)
            assert np.isfinite(depth).all() and np.isfinite(flip).all()
            temporary = path.with_suffix('.pending.npz')
            np.savez_compressed(temporary, depth_m=depth, flip_m=flip, confidence=confidence, image_index=index, checkpoint_sha256=identities['checkpoint_sha256'])
            temporary.replace(path)
            if (number + 1) % 50 == 0:
                print(f'Development cache {number+1}/795, median two-pass inference {np.median(latencies):.3f}s', flush=True)
    manifest['completed_predictions'] = len(list(output.glob('prediction_*.npz')))
    manifest['median_two_pass_inference_s'] = float(np.median(latencies)) if latencies else manifest.get('median_two_pass_inference_s')
    manifest['model_shape_constraints'] = adapter.model.shape_constraints
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({key: manifest[key] for key in ['completed_predictions', 'median_two_pass_inference_s']}, indent=2))


if __name__ == '__main__':
    main()
