"""Run a pinned Small image model on the declared NYUv2 dev or standard test split."""

import argparse
import importlib.metadata
import json
import platform
import subprocess
from time import perf_counter

import numpy as np
from PIL import Image

from run_stereo_baseline import ROOT, read_yaml, repo_path, checksum, write_json, write_csv
from p3_depth.models.runtime import ImageModel
from p3_depth.evaluation.benchmark import select_nyu, score_raw, summarize


def implementation_hashes():
    paths = ['scripts/run_nyuv2.py', 'scripts/run_stereo_baseline.py', 'src/p3_depth/models/runtime.py',
             'src/p3_depth/evaluation/benchmark.py', 'src/p3_depth/evaluation/depth.py',
             'src/p3_depth/data/nyuv2.py', 'environments/model_assets.json']
    return {name: checksum(ROOT / name) for name in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=['dav2_relative', 'dav2_metric', 'unidepth'])
    parser.add_argument('--subset', choices=['dev', 'test'], default='dev')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--precision', choices=['fp32', 'fp16'], default='fp32')
    parser.add_argument('--input-size', type=int, default=434, help='DA V2 short side; 434 matches UniDepth level=0 encoder shape for 480x640')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--paths-config', default='configs/paths.local.yaml')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    import torch
    packages = {k: importlib.metadata.version(k) for k in ('torch','torchvision','numpy','Pillow','opencv-python')}
    settings = read_yaml(repo_path(args.paths_config))
    manifest, rows = select_nyu(ROOT, args.subset, args.limit)
    output = repo_path(args.output_dir)
    config = {k: v for k, v in vars(args).items() if k not in ('resume', 'output_dir')}
    identity = {'config': config, 'manifest_sha256': checksum(manifest), 'sample_ids': [r['sample_id'] for r in rows],
                'implementation_sha256': implementation_hashes(), 'packages': packages, 'python': platform.python_version()}
    if output.exists():
        if not args.resume:
            raise FileExistsError(f'Use a new directory or explicit --resume: {output}')
        if json.loads((output / 'identity.json').read_text(encoding='utf-8')) != identity:
            raise ValueError('Resume requires unchanged configuration, manifest and implementation')
    else:
        if args.resume:
            raise FileNotFoundError('No existing run to resume')
        output.mkdir(parents=True)
        write_json(output / 'identity.json', identity)
    for name in ('predictions', 'masks', 'records', 'visualizations'):
        (output / name).mkdir(exist_ok=True)
    root = repo_path(settings['data_roots']['nyuv2'])
    model = ImageModel(args.model, ROOT, settings, args.device, args.input_size, args.precision, args.threads)
    previous_meta = json.loads((output / 'metadata.json').read_text(encoding='utf-8')) if (output / 'metadata.json').exists() else {}
    if previous_meta and previous_meta['model']['device_name'] != model.metadata['device_name']:
        raise ValueError('Resume requires the same device')
    warm_path = root / 'processed/rgb/nyuv2_0003.png'
    with Image.open(warm_path) as image:
        model(np.array(image.convert('RGB')))
    if args.device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    write_json(output / 'metadata.json', {'status': 'running', 'model': model.metadata, 'identity': identity,
        'code_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        'working_tree_status': subprocess.check_output(['git','status','--porcelain'], cwd=ROOT, text=True).strip(),
        'python': platform.python_version(), 'packages': packages,
        'scope': 'standard_test_654' if args.subset == 'test' and len(rows) == 654 else f'{args.subset}_smoke_{len(rows)}',
        'warmup': 'one train-side nyuv2_0003 image, excluded from timing',
        'timing': 'synchronized wall time, RGB numpy to original-size float32 depth; excludes model load, IO and evaluation',
        'upstream_preprocessing': 'official normalization, resize and original-shape restoration',
        'comparison_note': 'verify encoder shapes before comparing resolutions; UniDepth predicts intrinsics; no GT supplied to inference'})
    records = []
    for index, row in enumerate(rows):
        sample = row['sample_id']
        rgb_path, gt_path = (root / row['rgb_path']).resolve(), (root / row['depth_gt_path']).resolve()
        if not all(p.is_relative_to(root) for p in (rgb_path, gt_path)):
            raise ValueError('Manifest path escapes data root')
        rp = output / 'records' / f'{sample}.json'
        pp, mp = output / 'predictions' / f'{sample}.npy', output / 'masks' / f'{sample}.npy'
        if rp.exists():
            record = json.loads(rp.read_text(encoding='utf-8'))
            for path, key in ((pp,'prediction_sha256'), (mp,'mask_sha256'), (rgb_path,'rgb_sha256'), (gt_path,'gt_sha256')):
                if checksum(path) != record[key]:
                    raise ValueError(f'Resume input/output hash mismatch: {path}')
        else:
            with Image.open(rgb_path) as image:
                rgb = np.array(image.convert('RGB'))
            if args.device == 'cuda':
                torch.cuda.synchronize()
            start = perf_counter()
            raw = model(rgb)
            if args.device == 'cuda':
                torch.cuda.synchronize()
            elapsed = perf_counter() - start
            valid = np.isfinite(raw)
            if model.asset['prediction_type'] == 'metric_depth':
                valid &= raw > 0
            # Preserve model raw values; evaluation creates its own aligned copy.
            np.save(pp, raw)
            np.save(mp, valid)
            gt = np.load(gt_path, allow_pickle=False)
            scores, transform = score_raw(raw, gt, valid, model.asset)
            record = {'sample_id': sample, 'split': row['split'], 'inference_seconds': elapsed, 'metrics': scores,
                      'transformation': transform, 'prediction_sha256': checksum(pp), 'mask_sha256': checksum(mp),
                      'rgb_sha256': checksum(rgb_path), 'gt_sha256': checksum(gt_path),
                      'output_shape_hw': list(raw.shape), 'rgb_path': row['rgb_path'], 'gt_path': row['depth_gt_path']}
            write_json(rp, record)
        records.append(record)
        if index == 0 or (index + 1) % 25 == 0 or index + 1 == len(rows):
            print(f'{args.model} {index+1}/{len(rows)} {sample}: coverage={record["metrics"]["coverage"]:.3f}, RMSE={record["metrics"]["depth_on_valid_predictions"]["rmse_m"]}', flush=True)
    summary = summarize(records)
    write_json(output / 'summary.json', summary)
    write_csv(output / 'per_image.csv', [{'sample_id': r['sample_id'], 'split': r['split'], 'coverage': r['metrics']['coverage'],
        **r['metrics']['depth_on_valid_predictions'], 'seconds': r['inference_seconds']} for r in records])
    write_csv(output / 'predictions.csv', [{'sample_id': r['sample_id'], 'prediction_path': f'predictions/{r["sample_id"]}.npy',
        'mask_path': f'masks/{r["sample_id"]}.npy'} for r in records])
    metadata = json.loads((output / 'metadata.json').read_text(encoding='utf-8'))
    metadata.update(status='complete', model=model.metadata,
                    peak_cuda_allocated_bytes=max(previous_meta.get('peak_cuda_allocated_bytes') or 0, torch.cuda.max_memory_allocated()) if args.device == 'cuda' else None,
                    peak_cuda_reserved_bytes=max(previous_meta.get('peak_cuda_reserved_bytes') or 0, torch.cuda.max_memory_reserved()) if args.device == 'cuda' else None,
                    resumed_previous_run=bool(previous_meta))
    write_json(output / 'metadata.json', metadata)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
