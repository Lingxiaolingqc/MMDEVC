"""Three independent seeds with unchanged fitting/validation rules and full traces."""
import json
import time
import traceback
import numpy as np
import torch
from .model import make_model
from .data import load_pack
from .common import check_freeze
from .preflight import buffers, same_buffers
from p3_depth.residual_inputs.evaluation import batch, evaluate
from p3_depth.residual.model import corrected_depth, loss_function
from p3_depth.residual.job_schedule import segments, schedule_steps
from p3_depth.residual.training import atomic_save
from p3_depth.residual.evaluation import save_evaluation
from p3_depth.residual.records import sha, identity, write_json, append_json, now, remove_completed_resume


def checked_evaluation(model, cfg, pack, features, internal, detailed=False):
    before = buffers(model)
    result = evaluate(model, cfg, pack, features, internal, pack['original'], pack['validation'], 8, detailed)
    assert same_buffers(before, model), 'Validation changed training statistics'
    return result


def train(root, study, name, seed, resume=False):
    frozen = check_freeze(root, study)
    out = root/study['output']; run = out/'runs'/f'{name}_seed{seed}'
    run.mkdir(parents=True, exist_ok=True)
    if (run/'result.json').exists():
        result = json.loads((run/'result.json').read_text())
        assert result['design_sha256'] == sha(out/'design_frozen.json')
        assert all(sha(run/name) == value for name, value in result['checkpoint_sha256'].items())
        print(f'PASS already complete seed{seed}', flush=True)
        return result
    if (run/'config.json').exists() and not resume:
        raise FileExistsError('Existing run requires explicit --resume')
    try:
        torch.set_num_threads(4); torch.manual_seed(seed); np.random.seed(seed)
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        pack, features, internal = load_pack(root, study, name)
        cfg, settings = study['candidates'][name], study['training']
        model = make_model(features.shape[1], cfg, internal).cuda()
        config = {'seed': seed, 'candidate': cfg, 'training': settings,
                  'fit_ids': [pack['ids'][i] for i in pack['fitting']],
                  'validation_ids': [pack['ids'][i] for i in pack['validation']],
                  'parameters': sum(p.numel() for p in model.parameters()),
                  'design_sha256': sha(out/'design_frozen.json'), 'feature_hw': study['feature_hw'],
                  'source_sha256': frozen['source_sha256'], 'cache_sha256': pack['metadata']['cache_sha256'],
                  'feature_metadata_sha256': sha(out/'features/metadata.json'),
                  'validation_bn_policy': 'eval mode, exact buffer identity asserted at every evaluation'}
        assert config['parameters'] == frozen['parameters'][name]
        if (run/'config.json').exists():
            assert json.loads((run/'config.json').read_text()) == config
        else:
            write_json(run/'config.json', config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, schedule_steps(pack['fitting'], settings), eta_min=settings['minimum_learning_rate'])
        scaler = torch.amp.GradScaler('cuda', enabled=True)
        history = []; best = float('inf'); best_epoch = 0; stale = 0; completed = 0; updates = 0; exposures = 0; skips = 0
        if resume and (run/'resume.pt').exists():
            saved = torch.load(run/'resume.pt', map_location='cpu', weights_only=False)
            assert saved['config_hash'] == identity(config)
            model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler']); scaler.load_state_dict(saved['scaler'])
            history, best, best_epoch, stale, completed, updates, exposures, skips = [saved[k] for k in
                ('history', 'best', 'best_epoch', 'stale', 'epoch', 'updates', 'exposures', 'skips')]
            torch.set_rng_state(saved['rng_cpu']); torch.cuda.set_rng_state_all(saved['rng_cuda'])
            append_json(run/'events.jsonl', {'event': 'resume', 'epoch': completed})
        elif resume and (run/'history.json').exists():
            raise ValueError('History exists without resume state')
        previous = history[-1]['elapsed_seconds'] if history else 0.
        tick = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
        write_json(run/'status.json', {'status': 'running', 'seed': seed, 'at': now()})
        if not history:
            initial, _ = checked_evaluation(model, cfg, pack, features, internal)
            write_json(run/'initial_metrics.json', initial)
            print('INITIAL '+str(seed)+' '+json.dumps(initial), flush=True)
        for epoch, batches in segments(pack['fitting'], seed, settings):
            if epoch <= completed:
                continue
            # A resumed final epoch awaiting scoring must not trigger extra training.
            if completed >= settings['minimum_epochs'] and stale >= settings['patience_evaluations']:
                break
            model.train(); total = 0.; seen = 0
            for selected in batches:
                inputs, latent, bases, truths, masks, bands = batch(pack, features, internal, pack['original'], selected)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.float16):
                    residual = model(inputs, latent)
                prediction = corrected_depth(bases, residual, 'additive')
                loss = loss_function(prediction, truths, masks)
                if not torch.isfinite(loss) or not torch.isfinite(prediction).all():
                    raise ValueError('Nonfinite prediction or loss')
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['clip_gradient_norm'])
                if not torch.isfinite(norm):
                    skips += 1
                    append_json(run/'events.jsonl', {'event': 'AMP overflow; step skipped', 'update': updates+1, 'scale': scaler.get_scale()})
                scaler.step(optimizer); scaler.update(); scheduler.step()
                if scaler.get_scale() < 1:
                    raise ValueError('AMP scale collapsed; preserve failure, do not silently change protocol')
                total += float(loss.detach())*len(selected); seen += len(selected)
                updates += 1; exposures += len(selected)
            row = {'epoch': epoch, 'train_loss': total/seen, 'updates_attempted': updates,
                   'image_exposures': exposures, 'amp_skipped_updates': skips, 'amp_scale': scaler.get_scale(),
                   'learning_rate': optimizer.param_groups[0]['lr'], 'elapsed_seconds': previous+time.perf_counter()-tick}
            if epoch % settings['evaluate_every_epochs'] == 0 or epoch == settings['max_epochs']:
                measured, _ = checked_evaluation(model, cfg, pack, features, internal)
                row['validation'] = measured
                if measured['abs_rel'] < best:
                    best, best_epoch, stale = measured['abs_rel'], epoch, 0
                    atomic_save(run/'best.pt', {'model': model.state_dict(), 'config': config, 'epoch': epoch, 'selection_abs_rel': best})
                else:
                    stale += 1
            history.append(row); write_json(run/'history.json', history)
            atomic_save(run/'resume.pt', {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'config_hash': identity(config),
                'history': history, 'best': best, 'best_epoch': best_epoch, 'stale': stale,
                'epoch': epoch, 'updates': updates, 'exposures': exposures, 'skips': skips,
                'rng_cpu': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all()})
            append_json(run/'epochs.jsonl', row)
            print(json.dumps({'seed': seed, **row}), flush=True)
            if epoch >= settings['minimum_epochs'] and stale >= settings['patience_evaluations']:
                break
        atomic_save(run/'final.pt', {'model': model.state_dict(), 'config': config, 'epoch': history[-1]['epoch'], 'selection': 'last iterate'})
        final_timing, final_rows = checked_evaluation(model, cfg, pack, features, internal, True)
        final_metrics = save_evaluation(run/'final_validation', final_rows)
        model.load_state_dict(torch.load(run/'best.pt', map_location='cpu', weights_only=False)['model'])
        best_timing, rows = checked_evaluation(model, cfg, pack, features, internal, True)
        best_metrics = save_evaluation(run/'validation', rows)
        assert abs(best_metrics['abs_rel']-best) < 1e-7
        assert abs(final_metrics['abs_rel']-history[-1]['validation']['abs_rel']) < 1e-7
        result = {'status': 'complete', 'seed': seed, 'method': name, 'parameters': config['parameters'],
                  'design_sha256': config['design_sha256'], 'best_epoch': best_epoch,
                  'completed_epochs': history[-1]['epoch'], 'updates_attempted': updates, 'image_exposures': exposures,
                  'amp_skipped_updates': skips, 'validation': best_metrics, 'final_validation': final_metrics,
                  'elapsed_seconds': previous+time.perf_counter()-tick,
                  'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
                  'refiner_seconds_per_image': best_timing['refiner_seconds_per_image'],
                  'timing_scope': 'Cached refiner forward+restore; training elapsed includes validation, checkpoint IO and final detailed scoring; excludes foundation/cache loading',
                  'validation_bn_unchanged_all_evaluations': True,
                  'checkpoint_sha256': {n: sha(run/n) for n in ('best.pt', 'final.pt')},
                  'best_checkpoint_sha256': sha(run/'best.pt'), 'finished_at': now()}
        write_json(run/'result.json', result); write_json(run/'status.json', {'status': 'complete', 'at': now()})
        append_json(out/'runs.jsonl', result)
        remove_completed_resume(out, run)
        print('PASS '+json.dumps(result), flush=True)
        return result
    except BaseException as exc:
        failure = {'status': 'failed', 'seed': seed, 'at': now(), 'error': repr(exc), 'traceback': traceback.format_exc()}
        write_json(run/'status.json', failure); append_json(out/'failures.jsonl', failure)
        raise
