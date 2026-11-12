"""Fit a small residual refiner on scene-disjoint NYU development images."""
import json
from pathlib import Path
import time

import cv2
import h5py
import numpy as np
import torch
from torch.nn import functional as F

from depth_project.metrics import average_image_metrics, evaluate_nyu
from depth_project.refinement import refine_metric
from depth_project.residual_refiner import ResidualRefiner, make_features, apply_residual
from depth_project.records import run_provenance


def base_prediction(sample, config, rgb):
    return refine_metric(rgb, sample['depth_m'], sample['flip_m'], config)


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / 'runs/metric-refiner'
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'training.json').exists():
        raise FileExistsError('Training run already complete')
    manifest = json.loads((root / 'runs/nyu-development/manifest.json').read_text())
    search = json.loads((root / 'runs/metric-search/simple-search.json').read_text())
    config = search['best_simple']['config']
    seed = 320112
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    features = []
    bases = []
    truths = []
    validation = []
    fit_ids = set(manifest['fit_ids'])
    with h5py.File(root / 'data/nyu/nyu_depth_v2_labeled.mat') as dataset:
        for index in manifest['fit_ids'] + manifest['validation_ids']:
            with np.load(root / f'runs/nyu-development/prediction_{index:04d}.npz') as source:
                sample = {key: source[key] for key in ['depth_m', 'flip_m', 'confidence']}
            rgb = np.ascontiguousarray(dataset['images'][index].transpose(2, 1, 0))
            truth = dataset['depths'][index].T
            base = base_prediction(sample, config, rgb)
            feature = make_features(rgb, sample['depth_m'], sample['flip_m'], sample['confidence'], base)
            if index in fit_ids:
                features.append(feature.astype(np.float16))
                bases.append(cv2.resize(base, (320, 240), interpolation=cv2.INTER_LINEAR))
                truths.append(cv2.resize(truth, (320, 240), interpolation=cv2.INTER_NEAREST))
            else:
                validation.append((index, feature, base, truth))
    features = torch.from_numpy(np.stack(features)).cuda()
    bases = torch.from_numpy(np.stack(bases)[:, None]).cuda()
    truths = torch.from_numpy(np.stack(truths)[:, None]).cuda()
    mask = torch.zeros_like(truths, dtype=torch.bool)
    mask[:, :, 23:235, 21:300] = True
    mask &= torch.isfinite(truths) & (truths > .001) & (truths < 10)
    denominator = torch.where(mask, truths, 1.)
    model = ResidualRefiner().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 80, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda')
    history = []
    best_score = search['best_simple']['abs_rel']
    best_epoch = 0
    torch.save(model.state_dict(), output / 'best.pt')
    start = time.perf_counter()
    for epoch in range(1, 81):
        model.train()
        order = torch.randperm(len(features), device='cuda')
        losses = []
        for indices in order.split(8):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                correction = model(features[indices].float())
                prediction = (bases[indices] * torch.exp(correction.float())).clamp(.001, 10)
                relative = torch.abs(prediction - truths[indices]) / denominator[indices]
                loss = ((relative * mask[indices]).sum(dim=(1, 2, 3)) / mask[indices].sum(dim=(1, 2, 3))).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        scheduler.step()
        row = {'epoch': epoch, 'fit_abs_rel_low_resolution': float(np.mean(losses))}
        if epoch == 1 or epoch % 2 == 0:
            model.eval()
            rows = []
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                for index, feature, base, truth in validation:
                    residual = model(torch.from_numpy(feature[None]).cuda())[0, 0].float().cpu().numpy()
                    prediction = apply_residual(base, residual)
                    rows.append({'image_id': index, **evaluate_nyu(prediction, truth)})
            row['validation'] = average_image_metrics(rows)
            if row['validation']['abs_rel'] < best_score:
                best_score = row['validation']['abs_rel']
                best_epoch = epoch
                torch.save(model.state_dict(), output / 'best.pt')
            row['best_epoch'] = best_epoch
            row['elapsed_s'] = time.perf_counter() - start
            print(json.dumps(row), flush=True)
        history.append(row)
        (output / 'progress.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        if epoch >= 24 and epoch - best_epoch >= 20:
            break
    result = {'base_config': config, 'feature_hw': [240, 320], 'seed': seed, 'fit_ids': manifest['fit_ids'], 'validation_ids': manifest['validation_ids'], 'parameter_count': sum(parameter.numel() for parameter in model.parameters()), 'best_epoch': best_epoch, 'best_validation_abs_rel': best_score, 'history': history, 'training_s': time.perf_counter() - start, 'provenance': run_provenance(root), 'setting': 'NYU scene-disjoint supervised development adaptation of frozen UniDepthV2 ViT-S outputs'}
    (output / 'training.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (root / 'results/metric-refiner-training.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('BEST REFINER', best_epoch, best_score, flush=True)


if __name__ == '__main__':
    main()
