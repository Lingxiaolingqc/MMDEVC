"""Immutable follow-up specifications and scene-nested fitting subsets."""
import json
from pathlib import Path
import re
import numpy as np
from .model import candidate
from .records import identity, sha, write_json, now


PHASES = {'confirmation', 'combination', 'ablation', 'data_size', 'resolution', 'refit'}


def nested_scene_subsets(fitting_ids, scenes_by_id, seed=320118, fractions=(.1, .25, .5, 1.)):
    """One fixed scene ordering; closest nonempty prefix, never sampled validation."""
    groups = {}
    for index in fitting_ids:
        groups.setdefault(scenes_by_id[index], []).append(index)
    order = np.random.default_rng(seed).permutation(sorted(groups)).tolist()
    cumulative = np.cumsum([len(groups[name]) for name in order])
    subsets = []
    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError('Fitting fraction must be in (0,1]')
        end = int(np.argmin(np.abs(cumulative - len(fitting_ids)*fraction))) + 1
        selected = set(index for name in order[:end] for index in groups[name])
        subsets.append({'requested_fraction': fraction, 'actual_images': len(selected),
                        'actual_fraction': len(selected)/len(fitting_ids), 'scene_count': end,
                        'fit_ids': [i for i in fitting_ids if i in selected], 'scenes': order[:end]})
    if any(not set(a['fit_ids']).issubset(b['fit_ids']) for a, b in zip(subsets, subsets[1:])):
        raise ValueError('Requested fractions are not nested')
    return {'ordering_seed': seed, 'scene_order': order, 'subsets': subsets,
            'selection': 'nearest scene-prefix size to target; no scores or GT values used'}


def validate_job(job, split):
    if job['phase'] not in PHASES or not re.fullmatch(r'[A-Za-z0-9_-]+', job['run_id']):
        raise ValueError('Invalid phase/run ID')
    fits, vals = job['fit_ids'], job['validation_ids']
    if not fits or len(set(fits)) != len(fits) or len(set(vals)) != len(vals) or set(fits) & set(vals):
        raise ValueError('Duplicate/empty/overlapping data partition')
    tests = set(split['test_ids'])
    if tests & (set(fits) | set(vals)):
        raise ValueError('Test IDs cannot enter training or selection')
    if job['phase'] == 'refit':
        if set(fits) != set(split['fit_ids']+split['validation_ids']) or vals:
            raise ValueError('Final refit uses exactly all development IDs and no validation selection')
        if job['selection_point'] != 'fixed_final' or not job.get('frozen_selection_sha256'):
            raise ValueError('Final refit needs a frozen specification and final-iterate selection')
    else:
        if not set(fits).issubset(split['fit_ids']) or vals != split['validation_ids']:
            raise ValueError('Development follow-ups must preserve the fitting/validation boundary')
        if job['phase'] != 'data_size' and fits != split['fit_ids']:
            raise ValueError('Only a declared data-size experiment may reduce the fitting set')
    if job['feature_hw'] not in ([240, 320], [480, 640]):
        raise ValueError('Unexpected refiner resolution')
    if job['phase'] == 'data_size':
        if job['training'].get('max_steps', 0) <= 0 or job['selection_point'] != 'fixed_final':
            raise ValueError('Data-size control uses an explicit matched step budget and final iterate')
    elif job['training'].get('max_steps'):
        raise ValueError('Only the declared data-size study uses step-based training')
    if job['training']['max_epochs'] <= 0 or job['training']['batch_size'] <= 0:
        raise ValueError('Invalid training budget')
    cfg = job['candidate']
    if cfg['base'] not in ('original', 'scale', 'flip', 'flip_scale') or cfg['output'] not in ('additive', 'log', 'bounded_log'):
        raise ValueError('Invalid base/output factor')
    if cfg['loss'] not in ('abs_rel', 'gradient', 'boundary') or cfg['architecture'] not in ('cnn', 'mlp', 'unet', 'legacy_unet'):
        raise ValueError('Invalid architecture/loss factor')
    if cfg.get('internal_mode', 'true') not in ('true', 'zeros'):
        raise ValueError('Unknown internal feature control')
    return job


def build_job(study, split, name, phase, seed, run_id, *, overrides=None, fitting_ids=None,
              feature_hw=None, training_overrides=None, frozen_selection_sha256=None):
    cfg = candidate(name)
    cfg.update(overrides or {})
    training = {**study['training'], **(training_overrides or {})}
    job = {'schema': 1, 'run_id': run_id, 'phase': phase, 'seed': int(seed),
           'candidate': cfg, 'fit_ids': list(fitting_ids if fitting_ids is not None else split['fit_ids']),
           'validation_ids': [] if phase == 'refit' else list(split['validation_ids']),
           'feature_hw': list(feature_hw or study['feature_hw']), 'training': training,
           'selection_point': 'fixed_final' if phase in ('data_size', 'refit') else 'best_validation',
           'frozen_selection_sha256': frozen_selection_sha256,
           'test_history': study['protocol']['test_history']}
    return validate_job(job, split)


def save_job(root, study, job):
    split = json.loads((root / study['split']).read_text(encoding='utf-8'))
    validate_job(job, split)
    if job['phase'] == 'refit':
        path = root / study['output'] / 'frozen_selection.json'
        if not path.exists() or sha(path) != job['frozen_selection_sha256']:
            raise ValueError('The recorded final selection is absent or changed')
        from .selection import load_selection,check_refit_spec
        frozen,_=load_selection(root,study)
        check_refit_spec(job,frozen)
    record = {'created_at': now(), 'job': job, 'job_sha256': identity(job),
              'split_manifest_sha256': sha(root / study['split']),
              'dataset_sha256': split['dataset_sha256']}
    path = root / study['output'] / 'jobs' / (job['run_id'] + '.json')
    if path.exists():
        previous = json.loads(path.read_text(encoding='utf-8'))
        if previous['job_sha256'] != record['job_sha256'] or previous['split_manifest_sha256'] != record['split_manifest_sha256']:
            raise ValueError('Immutable job already exists with different settings')
        return path
    write_json(path, record)
    return path


def read_job(root, study, path):
    record = json.loads(Path(path).read_text(encoding='utf-8'))
    if identity(record['job']) != record['job_sha256'] or sha(root / study['split']) != record['split_manifest_sha256']:
        raise ValueError('Job or split identity changed')
    split = json.loads((root / study['split']).read_text(encoding='utf-8'))
    if record['dataset_sha256'] != split['dataset_sha256']:
        raise ValueError('Data identity changed')
    validate_job(record['job'], split)
    if record['job']['phase'] == 'refit':
        frozen = root / study['output'] / 'frozen_selection.json'
        if not frozen.exists() or sha(frozen) != record['job']['frozen_selection_sha256']:
            raise ValueError('Refit selection freeze changed')
        from .selection import load_selection,check_refit_spec
        selection,_=load_selection(root,study)
        check_refit_spec(record['job'],selection)
    return record
