"""Study identity and immutable provenance for the input extension."""
import json
from pathlib import Path
from p3_depth.residual.records import sha, identity, write_json, native_path, snapshot as parent_snapshot

def load(root):
    study=json.loads((root/'configs/residual_inputs_study.json').read_text(encoding='utf8'))
    parent=json.loads((root/study['parent_config']).read_text(encoding='utf8'))
    return study,parent

def snapshot(root,output):
    record=parent_snapshot(root,output)
    paths=[root/'configs/residual_inputs_study.json',root/'docs/residual_inputs_plan_20261119.md',
           *sorted((root/'src/p3_depth/residual_inputs').glob('*.py')),
           *sorted((root/'scripts/part3_improvement').glob('*input_study*.py')),root/'scripts/part3_improvement/analyze_input_test.py']
    for path in paths:
        name=path.relative_to(root).as_posix();target=native_path(output/'source_at_run'/name)
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
        record['source_sha256'][name]=sha(path)
    write_json(output/'provenance.json',record)
    return record

def candidate(study,name):
    value=dict(name=name,base='original',inputs='frozen_features',architecture='cnn',output='additive',loss='abs_rel')
    value.update(study['candidates'][name]);return value

def save_job(root,study,name,seed,phase='screen',epochs=None,smoke=False):
    if epochs is not None and phase!='refit' and not smoke:raise ValueError('Epoch overrides are reserved for frozen refits')
    split=json.loads((root/study['split']).read_text(encoding='utf8'))
    fits=split['fit_ids'];vals=split['validation_ids'];train=study['training'].copy()
    if phase=='refit':
        frozen=json.loads((root/study['output']/'frozen_selection.json').read_text(encoding='utf8'))
        choice=frozen['methods'][name];assert epochs==choice['epochs'] and seed in study['confirmation_seeds']
        fits=sorted(fits+vals);vals=[];train.update(max_epochs=epochs,schedule_epochs=80)
    if smoke:fits=fits[:16];vals=vals[:4];train.update(max_epochs=2,minimum_epochs=2,evaluate_every_epochs=1)
    job={'run_id':f'{phase}_{name}_seed{seed}'+('_smoke' if smoke else ''),'phase':phase,'seed':seed,
         'candidate':candidate(study,name),'training':train,'fit_ids':fits,'validation_ids':vals,
         'selection_point':'fixed_final' if phase=='refit' else 'best_validation','feature_hw':study['feature_hw'],
         'smoke':smoke,'split_sha256':sha(root/study['split'])}
    if phase=='refit':job['freeze_sha256']=sha(root/study['output']/'frozen_selection.json')
    record={'job':job,'job_sha256':identity(job)};path=root/study['output']/'jobs'/(job['run_id']+'.json')
    if path.exists():assert json.loads(path.read_text(encoding='utf8'))==record
    else:write_json(path,record)
    return path

def read_job(root,study,path):
    record=json.loads(Path(path).read_text(encoding='utf8'));job=record['job']
    assert identity(job)==record['job_sha256'] and job['split_sha256']==sha(root/study['split'])
    split=json.loads((root/study['split']).read_text(encoding='utf8'))
    assert not set(split['test_ids'])&(set(job['fit_ids'])|set(job['validation_ids']))
    assert not set(job['fit_ids'])&set(job['validation_ids'])
    if job['phase']=='refit':
        assert job['fit_ids']==sorted(split['fit_ids']+split['validation_ids']) and not job['validation_ids']
        assert job['freeze_sha256']==sha(root/study['output']/'frozen_selection.json')
    else:
        assert set(job['fit_ids'])<=set(split['fit_ids']) and set(job['validation_ids'])<=set(split['validation_ids'])
        if not job['smoke']:assert job['fit_ids']==split['fit_ids'] and job['validation_ids']==split['validation_ids']
    assert job['candidate']==candidate(study,job['candidate']['name'])
    return record
