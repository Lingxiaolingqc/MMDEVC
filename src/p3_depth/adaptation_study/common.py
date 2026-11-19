"""Immutable jobs and exact source provenance for the independent study."""
import json
from pathlib import Path
from p3_depth.residual.records import sha, identity, write_json, native_path, snapshot as parent_snapshot


def load(root):
    study = json.loads((root/'configs/fusion_lora_study.json').read_text(encoding='utf8'))
    parent = json.loads((root/study['parent_config']).read_text(encoding='utf8'))
    return study, parent


def snapshot(root, output):
    record = parent_snapshot(root, output)
    paths = [root/'configs/fusion_lora_study.json', root/'docs/fusion_lora_study_plan.md',
             *sorted((root/'src/p3_depth/adaptation_study').glob('*.py')),
             *sorted((root/'src/p3_depth/residual_inputs').glob('*.py')),
             *sorted((root/'scripts/part3_improvement').glob('*fusion_ablation*.py')),
             *sorted((root/'scripts/part3_improvement').glob('*lora_study*.py'))]
    for path in paths:
        name = path.relative_to(root).as_posix()
        target = native_path(output/'source_at_run'/name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        record['source_sha256'][name] = sha(path)
    write_json(output/'provenance.json', record)
    return record


def candidate(study, name):
    value = dict(name=name, base='original', inputs='frozen_features', architecture='cnn',
                 output='additive', loss='abs_rel')
    value.update(study['candidates'][name])
    return value


def size_ids(root, study, count):
    path = root/study['data_size']['source_design']
    record = json.loads(path.read_text(encoding='utf8'))
    assert record['split_sha256'] == sha(root/study['split'])
    subset = next(x for x in record['subset_design']['subsets'] if x['actual_images']==count)
    assert len(subset['fit_ids']) == count
    return subset['fit_ids'], sha(path)


def check_residual_freeze(root, study):
    path = root/study['output']/'residual_frozen.json'
    frozen = json.loads(path.read_text(encoding='utf8'))
    for name,digest in frozen['source_sha256'].items():
        assert sha(root/name) == digest, ('Residual numerical source changed after freeze',name)
    return frozen, sha(path)


def save_job(root, study, name, seed, phase='factorial', smoke=False, size=None):
    split = json.loads((root/study['split']).read_text(encoding='utf8'))
    fits, vals = split['fit_ids'], split['validation_ids']
    train = study['training'].copy()
    extra = {}
    if phase in ('data_size','refit'):
        frozen, freeze_hash = check_residual_freeze(root, study)
        assert name == frozen['method'] and not smoke
        extra['freeze_sha256'] = freeze_hash
    if phase == 'data_size':
        assert size in study['data_size']['counts']
        fits, extra['subset_design_sha256'] = size_ids(root, study, size)
        train.update(max_steps=study['data_size']['steps'],evaluate_every_steps=200)
    elif phase == 'refit':
        fits, vals = sorted(fits+vals), []
        train.update(max_epochs=frozen['epochs'],schedule_epochs=80)
    if smoke:
        fits, vals = fits[:16], vals[:4]
        train.update(max_epochs=2, minimum_epochs=2, evaluate_every_epochs=1)
    suffix = (f'_n{size}' if size is not None else '') + ('_smoke' if smoke else '')
    job = {'run_id':f'{phase}_{name}_seed{seed}' + suffix,
           'phase':phase, 'seed':seed, 'candidate':candidate(study,name), 'training':train,
           'fit_ids':fits, 'validation_ids':vals,
           'selection_point':'fixed_final' if phase in ('data_size','refit') else 'best_validation',
           'feature_hw':study['feature_hw'], 'smoke':smoke, 'split_sha256':sha(root/study['split']),**extra}
    record = {'job':job, 'job_sha256':identity(job)}
    path = root/study['output']/'jobs'/(job['run_id']+'.json')
    if path.exists():
        assert json.loads(path.read_text(encoding='utf8')) == record
    else:
        write_json(path, record)
    return path


def read_job(root, study, path):
    record = json.loads(Path(path).read_text(encoding='utf8'))
    job = record['job']
    assert identity(job) == record['job_sha256']
    assert job['split_sha256'] == sha(root/study['split'])
    split = json.loads((root/study['split']).read_text(encoding='utf8'))
    assert not set(split['test_ids']) & (set(job['fit_ids']) | set(job['validation_ids']))
    assert not set(job['fit_ids']) & set(job['validation_ids'])
    if job['phase'] in ('refit','data_size'):
        frozen, digest = check_residual_freeze(root, study)
        assert job['freeze_sha256'] == digest and job['candidate']['name'] == frozen['method']
    if job['phase'] == 'refit':
        assert job['fit_ids'] == sorted(split['fit_ids']+split['validation_ids']) and not job['validation_ids']
        assert job['freeze_sha256'] == sha(root/study['output']/'residual_frozen.json')
    elif job['phase'] == 'data_size':
        fits, digest = size_ids(root, study, len(job['fit_ids']))
        assert job['fit_ids'] == fits and job['subset_design_sha256'] == digest
        assert job['validation_ids'] == split['validation_ids']
        assert job['freeze_sha256'] == sha(root/study['output']/'residual_frozen.json')
    elif not job['smoke']:
        assert job['fit_ids'] == split['fit_ids'] and job['validation_ids'] == split['validation_ids']
    else:
        assert set(job['fit_ids']) <= set(split['fit_ids'])
        assert set(job['validation_ids']) <= set(split['validation_ids'])
    assert job['candidate'] == candidate(study, job['candidate']['name'])
    return record
