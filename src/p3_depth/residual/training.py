"""Bounded runs with full records, epoch-boundary resume, and selected-weight evaluation."""
import json
import math
import time
import traceback
import numpy as np
import torch
from p3_depth.data.nyuv2 import NYUv2Labeled
from .data import load_pack, calibration, base_for, features_for
from .model import candidate, Refiner, corrected_depth, loss_function
from .evaluation import score_image, save_evaluation
from .records import write_json, append_json, identity, snapshot, sha, now, remove_completed_resume


def atomic_save(path, state):
    temporary = path.with_name(path.name + '.pending')
    torch.save(state, temporary)
    temporary.replace(path)


def batch(pack, features, internal, base, positions):
    def tensor(value):
        return torch.from_numpy(np.ascontiguousarray(value[positions])).to('cuda')
    return (tensor(features), tensor(internal).float() if internal is not None else None,
            tensor(base)[:, None], tensor(pack['truth'])[:, None], tensor(pack['mask'])[:, None], tensor(pack['band'])[:, None])


def evaluate(model, config, pack, features, internal, base, positions, batch_size, detailed=False):
    model.eval()
    rows = []
    times = []
    with torch.inference_mode():
        for start in range(0, len(positions), batch_size):
            selected = positions[start:start+batch_size]
            inputs, latent, bases, truths, masks, _ = batch(pack, features, internal, base, selected)
            torch.cuda.synchronize()
            tick = time.perf_counter()
            with torch.autocast('cuda', dtype=torch.float16):
                residual = model(inputs, latent)
            prediction = corrected_depth(bases, residual, config['output'])
            torch.cuda.synchronize()
            times.append((time.perf_counter()-tick, len(selected)))
            if not bool(torch.isfinite(prediction).all()):
                raise ValueError('Nonfinite validation prediction')
            if detailed:
                array = prediction[:, 0].cpu().numpy()
                rows.extend(score_image(array[j], pack, i) for j, i in enumerate(selected))
            else:
                clipped, truth = prediction.clamp(.001, 10).double(), truths.double()
                safe = torch.where(masks, truth, 1.)
                count = masks.sum((1, 2, 3))
                difference = clipped - safe
                scores = torch.stack([((difference.abs()/safe)*masks).sum((1, 2, 3))/count,
                                      ((difference.square()*masks).sum((1, 2, 3))/count).sqrt(),
                                      ((torch.maximum(clipped/safe, safe/clipped) < 1.25)*masks).sum((1, 2, 3))/count], -1).cpu().numpy()
                rows.extend({'abs_rel': float(v[0]), 'rmse_m': float(v[1]), 'delta1': float(v[2])} for v in scores)
    mean = {key: float(np.mean([r[key] for r in rows])) for key in ('abs_rel', 'rmse_m', 'delta1')}
    mean['refiner_seconds_per_image'] = sum(x[0] for x in times)/sum(x[1] for x in times)
    return mean, rows


def preview(root, study, output, model, config, pack, features, internal, base, positions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Fixed before looking at candidate scores: first/middle/last validation samples.
    selected = [positions[i] for i in sorted({0, len(positions)//2, len(positions)-1})]
    with NYUv2Labeled(root / study['data']) as data, torch.inference_mode():
        for i in selected:
            inputs, latent, bases, _, _, _ = batch(pack, features, internal, base, [i])
            with torch.autocast('cuda', dtype=torch.float16):
                residual = model(inputs, latent)
            predicted = corrected_depth(bases, residual, config['output'])[0, 0].cpu().numpy()
            rgb = data.load(pack['ids'][i]+1).rgb
            fig, axes = plt.subplots(1, 5, figsize=(15, 3.6), constrained_layout=True)
            axes[0].imshow(rgb)
            for ax, values in zip(axes[1:4], [pack['truth'][i], base[i], predicted]):
                ax.imshow(values, vmin=0, vmax=10, cmap='viridis')
            difference = predicted - base[i]
            axes[4].imshow(difference, vmin=-.5, vmax=.5, cmap='coolwarm')
            for ax, title in zip(axes, ['RGB', 'GT (0-10m)', 'Base (0-10m)', 'Refined (0-10m)', 'Correction (-.5,+.5m)']):
                ax.set_title(title)
                ax.axis('off')
            fig.suptitle(f'{config["name"]} | nyuv2_{pack["ids"][i]+1:04d} | validation; fixed display limits')
            fig.savefig(output / f'nyuv2_{pack["ids"][i]+1:04d}.png', dpi=140)
            plt.close(fig)


def run_training(root, study, name, seed, smoke=False, resume=False, shared_pack=None):
    configuration = candidate(name)
    training = study['training'].copy()
    if smoke:
        training.update(max_epochs=2, evaluate_every_epochs=1, minimum_epochs=2)
    label = f'{name}_seed{seed}' + ('_smoke' if smoke else '')
    study_root = root / study['output']
    output = study_root / 'runs' / label
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / 'config.json'
    if config_path.exists() and not resume:
        raise FileExistsError(f'Run exists; explicit --resume required: {output}')
    if (output / 'result.json').exists():
        result = json.loads((output / 'result.json').read_text(encoding='utf-8'))
        if result['status'] == 'complete':
            if sha(output / 'best.pt') != result['best_checkpoint_sha256']:
                raise ValueError('Completed checkpoint was modified')
            print('PASS already complete '+label, flush=True)
            return result
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        pack = load_pack(root, study) if shared_pack is None else shared_pack
        fitted = calibration(root, study, pack)
        base = base_for(pack, configuration['base'], fitted)
        # Edge candidates are intentionally gated on a recorded fitting-only preparation.
        edge = None
        if configuration['inputs'] in ('rgb_sobel', 'rgb_canny', 'depth_sobel', 'depth_canny', 'joint_edges'):
            from .edges import load_edge_features
            edge = load_edge_features(root, study, pack)
        features, internal = features_for(pack, configuration, base, study['feature_hw'], edge)
        fits = pack['fitting'][:32] if smoke else pack['fitting']
        vals = pack['validation'][:8] if smoke else pack['validation']
        model = Refiner(features.shape[1], configuration['architecture'], internal.shape[1] if internal is not None else 0).cuda()
        config = {'candidate': configuration, 'training': training, 'seed': seed, 'smoke': smoke,
                  'fit_ids': [pack['ids'][i] for i in fits], 'validation_ids': [pack['ids'][i] for i in vals],
                  'cache_sha256': pack['metadata']['cache_sha256'], 'calibration': fitted,
                  'feature_hw': study['feature_hw'], 'input_channels': features.shape[1],
                  'internal_channels': internal.shape[1] if internal is not None else 0,
                  'parameters': sum(x.numel() for x in model.parameters()),
                  'data_order': 'numpy.default_rng(seed+epoch).permutation(fit_positions)',
                  'amp': True, 'cudnn_deterministic': True, 'deterministic_algorithms_enforced': False,
                  'training_prediction_clip': [.001, 10.], 'gt_in_inference': False}
        config_hash = identity(config)
        if config_path.exists():
            if identity(json.loads(config_path.read_text(encoding='utf-8'))) != config_hash:
                raise ValueError('Run configuration changed')
            previous = json.loads((output / 'provenance.json').read_text(encoding='utf-8'))
            for path, expected in previous['source_sha256'].items():
                if sha(root / path) != expected:
                    raise ValueError('Source changed; do not resume incompatible run: '+path)
        else:
            write_json(config_path, config)
            snapshot(root, output)
        optimizer = torch.optim.AdamW(model.parameters(), lr=training['learning_rate'], weight_decay=training['weight_decay'])
        steps_per_epoch = math.ceil(len(fits)/training['batch_size'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, training['max_epochs']*steps_per_epoch, eta_min=training['minimum_learning_rate'])
        scaler = torch.amp.GradScaler('cuda', enabled=True)
        history, best_score, best_epoch, start_epoch, stale = [], float('inf'), 0, 0, 0
        if resume and (output / 'resume.pt').exists():
            saved = torch.load(output / 'resume.pt', map_location='cpu', weights_only=False)
            if saved['config_hash'] != config_hash:
                raise ValueError('Resume identity mismatch')
            model.load_state_dict(saved['model'])
            optimizer.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler'])
            scaler.load_state_dict(saved['scaler'])
            history, best_score, best_epoch, start_epoch, stale = [saved[k] for k in ('history', 'best_score', 'best_epoch', 'epoch', 'stale')]
            torch.set_rng_state(saved['rng_cpu'])
            torch.cuda.set_rng_state_all(saved['rng_cuda'])
            write_json(output / 'history.json', history)
        elif resume and config_path.exists() and (output / 'history.json').exists():
            raise ValueError('History exists without resume checkpoint; use a new run')
        started = time.perf_counter()
        previous_seconds = history[-1]['elapsed_seconds'] if history else 0.
        torch.cuda.reset_peak_memory_stats()
        append_json(study_root / 'runs.jsonl', {'run': label, 'status': 'running', 'configuration_sha256': config_hash})
        write_json(output / 'status.json', {'status': 'running', 'started_at': now(), 'configuration_sha256': config_hash})
        if not history:
            initial, _ = evaluate(model, configuration, pack, features, internal, base, vals, training['batch_size'])
            write_json(output / 'initial_metrics.json', initial)
            print(f'INITIAL {label} {initial}', flush=True)
        for epoch in range(start_epoch, training['max_epochs']):
            model.train()
            order = np.random.default_rng(seed+epoch).permutation(fits).tolist()
            epoch_losses = []
            for offset in range(0, len(order), training['batch_size']):
                selected = order[offset:offset+training['batch_size']]
                inputs, latent, bases, truths, masks, bands = batch(pack, features, internal, base, selected)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.float16):
                    raw_residual = model(inputs, latent)
                prediction = corrected_depth(bases, raw_residual, configuration['output'])
                loss = loss_function(prediction, truths, masks, configuration['loss'], bands,
                                     training['gradient_loss_weight'], training['boundary_loss_weight'])
                if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(prediction).all()):
                    raise ValueError('Nonfinite training loss/prediction')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training['clip_gradient_norm'])
                if not bool(torch.isfinite(norm)):
                    # GradScaler may skip transient AMP overflow; retain an explicit event.
                    append_json(output / 'events.jsonl', {'epoch': epoch+1, 'batch_offset': offset, 'event': 'AMP overflow; optimizer step skipped'})
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                epoch_losses.append((float(loss.detach()), len(selected)))
            row = {'epoch': epoch+1, 'updates_attempted': (epoch+1)*steps_per_epoch,
                   'image_exposures': (epoch+1)*len(fits),
                   'train_loss': sum(x*n for x, n in epoch_losses)/len(fits),
                   'learning_rate': optimizer.param_groups[0]['lr'],
                   'elapsed_seconds': previous_seconds+time.perf_counter()-started}
            if (epoch+1) % training['evaluate_every_epochs'] == 0 or epoch+1 == training['max_epochs']:
                metrics, _ = evaluate(model, configuration, pack, features, internal, base, vals, training['batch_size'])
                row['validation'] = metrics
                if metrics['abs_rel'] < best_score:
                    best_score, best_epoch, stale = metrics['abs_rel'], epoch+1, 0
                    atomic_save(output / 'best.pt', {'model': model.state_dict(), 'config': config, 'epoch': best_epoch, 'selection_abs_rel': best_score})
                else:
                    stale += 1
            history.append(row)
            write_json(output / 'history.json', history)
            atomic_save(output / 'resume.pt', {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'config_hash': config_hash,
                        'history': history, 'best_score': best_score, 'best_epoch': best_epoch, 'epoch': epoch+1,
                        'stale': stale, 'rng_cpu': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all()})
            print(json.dumps({'run': label, **row}), flush=True)
            if epoch+1 >= training['minimum_epochs'] and stale >= training['patience_evaluations']:
                break
        atomic_save(output/'final.pt', {'model':model.state_dict(),'config':config,'epoch':len(history),'selection':'last training iterate'})
        model.load_state_dict(torch.load(output / 'best.pt', map_location='cpu', weights_only=False)['model'])
        metrics, rows = evaluate(model, configuration, pack, features, internal, base, vals, training['batch_size'], detailed=True)
        measured = save_evaluation(output / 'validation', rows)
        if abs(measured['abs_rel']-best_score) > 1e-7:
            raise ValueError('Selected checkpoint does not reproduce selection score')
        (output / 'visualizations').mkdir(exist_ok=True)
        preview(root, study, output / 'visualizations', model, configuration, pack, features, internal, base, vals)
        result = {'status': 'complete', 'run': label, 'configuration_sha256': config_hash,
                  'best_epoch': best_epoch, 'completed_epochs': len(history), 'validation': measured,
                  'parameters': config['parameters'], 'best_checkpoint_sha256': sha(output / 'best.pt'),
                  'final_checkpoint_sha256': sha(output/'final.pt'),
                  'elapsed_seconds': previous_seconds+time.perf_counter()-started,
                  'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
                  'refiner_seconds_per_image': metrics['refiner_seconds_per_image'],
                  'timing_scope': 'refiner GPU forward and restore only; frozen model/flip timings are in shared cache',
                  'smoke': smoke, 'training_finished_at': now()}
        write_json(output / 'result.json', result)
        write_json(output / 'status.json', {'status': 'complete', 'finished_at': now()})
        initial = json.loads((output / 'initial_metrics.json').read_text(encoding='utf-8'))
        delta = measured['abs_rel'] - initial['abs_rel']
        (output / 'review.md').write_text(f'# {label}\n\n'+('Engineering smoke only; not a formal result.\n\n' if smoke else '')+
            f'Validation AbsRel: base {initial["abs_rel"]:.8f} -> selected refiner {measured["abs_rel"]:.8f} (change {delta:+.8f}).\n\n'
            f'Selected epoch {best_epoch}; completed {len(history)}. Boundary macro F1 {measured["boundary_f1"]:.6f}. '
            'This single run is descriptive; seed confirmation and visual inspection are separate steps. '
            'Per-image scores, complete history, source and best weights are retained.\n', encoding='utf-8')
        append_json(study_root / 'runs.jsonl', {'run': label, 'status': 'complete', 'result': result})
        remove_completed_resume(study_root, output)
        print('PASS '+label+' '+json.dumps(measured), flush=True)
        return result
    except BaseException as exc:
        failed = {'status': 'failed', 'run': label, 'at': now(), 'error': repr(exc), 'traceback': traceback.format_exc()}
        write_json(output / 'status.json', failed)
        append_json(study_root / 'runs.jsonl', failed)
        raise


def evaluate_bases(root, study):
    from .records import snapshot
    output = root / study['output'] / 'bases'
    if (output / 'comparison.json').exists():
        raise FileExistsError('Baseline results already exist; inspect or use a new study')
    output.mkdir(parents=True, exist_ok=True)
    snapshot(root, output)
    pack = load_pack(root, study)
    fitted = calibration(root, study, pack)
    result = {}
    for label, name in [('B0', 'original'), ('B1', 'scale'), ('B2', 'flip'), ('B3', 'flip_scale')]:
        base = base_for(pack, name, fitted)
        rows = [score_image(base[i], pack, i) for i in pack['validation']]
        result[label] = save_evaluation(output / label, rows)
        print(label+' '+json.dumps(result[label]), flush=True)
    write_json(output / 'comparison.json', {'validation': result, 'calibration': fitted,
               'cache_sha256': pack['metadata']['cache_sha256'], 'test_used': False})
