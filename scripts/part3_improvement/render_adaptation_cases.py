"""Render matched validation depth/error/edge cases within one adaptation family."""
import argparse
import json
import sys
from pathlib import Path
import h5py
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from analyze_residual_study import read_csv
from benchmark_input_study import load_refiner
from p3_depth.adaptation_study.common import load
from p3_depth.adaptation_study.model import make_model
from p3_depth.adaptation_study.lora import inject, load_adapters, differentiable_infer
from p3_depth.models.runtime import ImageModel
from p3_depth.data.nyuv2 import NYUv2Labeled
from p3_depth.residual.deployment import image_inputs, predict_arrays
from p3_depth.residual_inputs.data import observable_inputs
from p3_depth.residual.model import corrected_depth
from p3_depth.residual.final_evaluation import scoring_pack
from p3_depth.residual.evaluation import score_image
from p3_depth.evaluation.boundary import boundary_masks
from p3_depth.residual.records import write_json, sha
from p3_depth.residual.locking import study_lock


def locate(out, name, seed):
    found = []
    for path in (out / 'followups').glob('*/config.json'):
        config = json.loads(path.read_text())
        if config['candidate']['name'] != name or config['seed'] != seed:
            continue
        result_path = path.parent / 'result.json'
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text())
        if result['smoke'] or result['phase'] not in ('factorial', 'control', 'zero_middle', 'search', 'confirm'):
            continue
        found.append(path.parent)
    assert len(found) == 1, (name, seed, found)
    return found[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=['residual', 'lora'], required=True)
    parser.add_argument('--before', default='S0')
    parser.add_argument('--after', required=True)
    parser.add_argument('--seed', type=int, default=320112)
    args = parser.parse_args()
    study, _ = load(ROOT)
    out = ROOT / study['output']
    old = ROOT / study['parent_output']
    assert args.after in (study['candidates'] if args.family == 'residual' else
                          __import__('p3_depth.adaptation_study.lora_training', fromlist=['candidates']).candidates(study))
    if args.family == 'lora':
        assert args.before == 'B0', 'No Residual-versus-LoRA comparison in this selection stage'
    else:
        assert args.before == 'B0' or args.before in study['candidates']
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with study_lock(out):
        render(args, study, out, old)


def render(args, study, out, old):
    runs = {}
    for name in (args.before, args.after):
        if name == 'B0':
            runs[name] = old / 'bases/B0'
        elif name == 'S0':
            runs[name] = old / 'runs' / f'frozen_features_seed{args.seed}'
        else:
            runs[name] = locate(out, name, args.seed)
    scores = {name: {int(r['source_index']): r for r in read_csv(run / ('per_image.csv' if name == 'B0' else 'validation/per_image.csv'))}
              for name, run in runs.items()}
    before, after = scores[args.before], scores[args.after]
    assert before.keys() == after.keys() and len(after) == 218
    ids = list(after)
    ranked = sorted(ids, key=lambda i: float(after[i]['abs_rel']) - float(before[i]['abs_rel']))
    choices = {'fixed_first': ids[0], 'fixed_middle': ids[len(ids)//2], 'fixed_last': ids[-1],
               'largest_gain': ranked[0], 'median_change': ranked[len(ranked)//2], 'largest_regression': ranked[-1],
               'largest_boundary_regression': min(ids, key=lambda i: float(after[i]['boundary_f1']) - float(before[i]['boundary_f1']))}
    destination = out / 'analysis/visuals' / f'{args.family}_{args.before}_to_{args.after}_seed{args.seed}'
    destination.mkdir(parents=True, exist_ok=True)
    stage = out / 'stage_cache/development'
    meta = json.loads((stage / 'metadata.json').read_text())
    positions = {index: i for i, index in enumerate(meta['spec']['ids'])}
    arrays = {name: np.load(stage / (name + '.npy'), mmap_mode='r') for name in ('init', 'mid', 'late')}
    edge = json.loads((old / 'edge_features/metadata.json').read_text())['calibration']
    models, configs, hashes = {}, {}, {}
    for name, run in runs.items():
        if name == 'B0':
            continue
        checkpoint = run / 'best.pt'
        hashes[checkpoint.relative_to(ROOT).as_posix()] = sha(checkpoint)
        record = torch.load(checkpoint, map_location='cpu', weights_only=False)
        cfg = record['config']
        if name == 'S0':
            models[name], configs[name] = load_refiner(checkpoint)
            continue
        configs[name] = cfg
        if args.family == 'lora':
            settings = yaml.safe_load((ROOT / 'configs/paths.local.yaml').read_text())
            adapter = ImageModel('unidepth', ROOT, settings, 'cuda', precision='fp16', threads=4)
            model = adapter.model
            targets = inject(model, cfg['candidate']['target'], cfg['candidate']['rank'])
            assert targets == cfg['target_paths']
            load_adapters(model, record['model'])
        else:
            model = make_model(4, cfg['candidate'], arrays).cuda().eval()
            model.load_state_dict(record['model'])
        models[name] = model
    checks = []
    with NYUv2Labeled(ROOT / study['data']) as source, h5py.File(old / 'cache/development/predictions.h5', 'r') as cache, torch.no_grad():
        for reason, index in choices.items():
            sample = source.load(index + 1)
            rgb, truth = sample.rgb, sample.depth_m
            group = cache[str(index)]
            base = group['original'][:]
            predictions = {}
            for name in runs:
                if name == 'B0':
                    predictions[name] = base
                    continue
                cfg, model = configs[name], models[name]
                if name == 'S0':
                    inputs = image_inputs(rgb, base, cfg['candidate'], cfg['feature_hw'], cfg['calibration'], internal=group['internal'][:])
                    predictions[name] = predict_arrays(model, cfg, *inputs)[0].cpu().numpy()
                elif args.family == 'lora':
                    inputs = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])).cuda()
                    predictions[name] = differentiable_infer(model, inputs)[0, 0].cpu().numpy()
                else:
                    features = observable_inputs(rgb, base, group['confidence'][:], cfg['candidate'], study['feature_hw'], edge)
                    pos = positions[index]
                    latents = {k: torch.from_numpy(np.array(v[pos:pos+1], copy=True)).float().cuda() for k, v in arrays.items() if k in cfg['candidate']['branches']}
                    with torch.autocast('cuda', dtype=torch.float16):
                        residual = model(torch.from_numpy(features[None]).cuda(), latents)
                    predictions[name] = corrected_depth(torch.from_numpy(base[None, None]).cuda(), residual, 'additive')[0, 0].cpu().numpy()
            pack = scoring_pack(sample, index)
            mask = pack['mask'][0]
            measured = {name: score_image(value, pack, 0) for name, value in predictions.items()}
            differences = {name: abs(measured[name]['abs_rel'] - float(scores[name][index]['abs_rel'])) for name in runs}
            assert max(differences.values()) < 1e-4, (index, differences)
            fig, axes = plt.subplots(3, 4, figsize=(14, 10), constrained_layout=True)
            axes[0, 0].imshow(rgb)
            axes[0, 0].set_title('RGB')
            values = [truth, predictions[args.before], predictions[args.after]]
            for ax, value, label in zip(axes[0, 1:], values, ['GT', args.before, args.after]):
                ax.imshow(value, vmin=0, vmax=10, cmap='viridis')
                ax.set_title(label + ' | 0-10 m')
            axes[1, 0].imshow(mask, cmap='gray')
            axes[1, 0].set_title('Course mask')
            axes[1, 1].text(.02, .5, f'{reason}\nnyuv2_{index+1:04d}\n' + '\n'.join(f'{n}: {measured[n]["abs_rel"]:.5f}' for n in runs), fontsize=11)
            for ax, value in zip(axes[1, 2:], values[1:]):
                error = np.where(mask, np.abs(np.clip(value, .001, 10)-truth)/np.maximum(truth, .001), np.nan)
                ax.imshow(error, vmin=0, vmax=.5, cmap='magma')
                ax.set_title('AbsRel error | 0-0.5')
            axes[2, 0].text(.02, .4, 'GT edges = cyan\nPredicted = magenta\nExact overlap = white\nMetric tolerance = 2 px', fontsize=11)
            _, _, gt_edge, _ = boundary_masks(truth, truth, mask, mask)
            axes[2, 1].imshow(gt_edge, cmap='gray')
            axes[2, 1].set_title('GT depth edges')
            for ax, value in zip(axes[2, 2:], values[1:]):
                clipped = np.clip(value, .001, 10).astype(np.float32)
                _, _, gt_edge, pred_edge = boundary_masks(clipped, truth, mask, np.isfinite(clipped) & (clipped > 0))
                overlay = np.repeat((rgb.mean(-1)/255*.25)[..., None], 3, -1)
                overlay[gt_edge] = [0, 1, 1]
                overlay[pred_edge] = [1, 0, 1]
                overlay[gt_edge & pred_edge] = [1, 1, 1]
                ax.imshow(overlay)
                ax.set_title('Depth edge agreement')
            for ax in axes.flat:
                ax.axis('off')
            fig.suptitle(f'Validation only | seed {args.seed} | fixed scales; selected cases do not represent the average')
            name = f'{reason}_nyuv2_{index+1:04d}.png'
            fig.savefig(destination / name, dpi=130)
            plt.close(fig)
            checks.append({'reason': reason, 'source_index': index, 'metrics': measured,
                           'saved_score_abs_difference': differences, 'figure': name, 'sha256': sha(destination / name)})
    write_json(destination / 'cases.json', {'status': 'complete', 'before': args.before, 'after': args.after,
        'seed': args.seed, 'family': args.family, 'checkpoints': hashes, 'cases': checks,
        'scope': '218-image validation; fixed first/middle/last, gain/median/regression by this seed, worst F1 regression; not representative sampling or a test set.'})
    print('PASS matched validation cases: ' + str(destination), flush=True)


if __name__ == '__main__':
    main()
