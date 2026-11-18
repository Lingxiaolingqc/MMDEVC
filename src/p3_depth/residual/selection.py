"""Freeze numerical methods and durations before final refitting/test inference."""
import json
from pathlib import Path
from .records import sha,write_json,now


NUMERICAL_SOURCES=['model.py','data.py','edges.py','job_inputs.py','job_schedule.py',
                   'training.py','followup.py','evaluation.py','deployment.py','final_evaluation.py','selection.py','jobs.py','cache.py']
SHARED_NUMERICAL_SOURCES=['src/p3_depth/data/nyuv2.py','src/p3_depth/models/runtime.py',
                         'src/p3_depth/evaluation/course.py','src/p3_depth/evaluation/boundary.py',
                         'src/p3_depth/evaluation/rgb_depth_edges.py']


def check_refit_spec(job,frozen):
    choices={row['run_id']:row for row in frozen['refits']}
    if job['run_id'] not in choices:raise ValueError('Refit run was not frozen')
    planned=choices[job['run_id']]
    for key in ('seed','candidate','training','feature_hw'):
        if job[key]!=planned[key]:raise ValueError('Refit differs from frozen '+key)
    if set(job['fit_ids'])!=set(frozen['development_ids']) or job['validation_ids'] or job['selection_point']!='fixed_final':
        raise ValueError('Refit partition/selection differs from freeze')


def load_selection(root,study,check_sources=True):
    path=root/study['output']/'frozen_selection.json'
    record=json.loads(path.read_text(encoding='utf-8'))
    if record['status']!='frozen_before_new_test_inference' or record['split_sha256']!=sha(root/study['split']):
        raise ValueError('Missing/changed final split freeze')
    if record['protocol']!=study['protocol']:raise ValueError('Final evaluation protocol changed')
    for name,digest in record['development_evidence_sha256'].items():
        if sha(root/name)!=digest:raise ValueError('Development evidence changed after freeze')
    if check_sources:
        for name,digest in record['numerical_source_sha256'].items():
            if sha(root/name)!=digest:raise ValueError('Numerical implementation changed after freeze: '+name)
    return record,sha(path)


def freeze_selection(root,study,proposal):
    output=root/study['output'];path=output/'frozen_selection.json'
    if path.exists():raise FileExistsError('Final selection already frozen; never silently replace it')
    if (output/'cache/test/metadata.json').exists():raise ValueError('New test inference already exists; cannot claim a prior freeze')
    split=json.loads((root/study['split']).read_text(encoding='utf-8'))
    evidence={}
    for relative in proposal['development_evidence']:
        source=(root/relative).resolve()
        if not source.is_relative_to(output.resolve()):raise ValueError('Development evidence must belong to this study')
        result=json.loads(source.read_text(encoding='utf-8'))
        if result.get('status')!='complete' or result.get('smoke') or (result.get('validation') or {}).get('images')!=218:
            raise ValueError('Incomplete/non-validation evidence for final selection')
        evidence[relative]=sha(source)
    if not evidence:raise ValueError('No completed development evidence')
    refits=[dict(row) for row in proposal['refits']]
    if not refits or len({x['run_id'] for x in refits})!=len(refits):raise ValueError('Missing/duplicate final run IDs')
    groups={}
    from .jobs import build_job
    for row in refits:
        if row['training'].get('max_steps') or row['training']['max_epochs']<=0:
            raise ValueError('Final duration must be a positive predeclared epoch count')
        groups.setdefault(row['method'],[]).append(row['seed'])
        job=build_job(study,split,'R0','refit',row['seed'],row['run_id'],overrides=row['candidate'],
                  fitting_ids=sorted(split['fit_ids']+split['validation_ids']),feature_hw=row['feature_hw'],
                  training_overrides=row['training'],frozen_selection_sha256='pending_freeze_identity')
        for key in ('candidate','training','feature_hw'):row[key]=job[key]
    if any(sorted(seeds)!=sorted(study['confirmation_seeds']) for seeds in groups.values()):
        raise ValueError('Final methods must use all three declared seeds')
    record={'schema':1,'status':'frozen_before_new_test_inference','recorded_at':now(),
            'development_ids':sorted(split['fit_ids']+split['validation_ids']),'test_ids':split['test_ids'],
            'split_sha256':sha(root/study['split']),'dataset_sha256':split['dataset_sha256'],
            'protocol':study['protocol'],'baseline_methods':['B0','B1','B2','B3'],
            'baseline_calibration':'Fit global scales only on all 795 development images; no test-GT fit',
            'refits':refits,'development_evidence_sha256':evidence,
            'rationale':proposal['rationale'],'duration_rule':proposal['duration_rule'],
            'test_history':study['protocol']['test_history'],
            'numerical_source_sha256':{name:sha(root/name) for name in [*[f'src/p3_depth/residual/{name}' for name in NUMERICAL_SOURCES],*SHARED_NUMERICAL_SOURCES]}}
    write_json(path,record)
    return record,sha(path)
