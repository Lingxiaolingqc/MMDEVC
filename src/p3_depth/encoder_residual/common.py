"""Immutable encoder study identity and audited decoder validation references."""
import json
import platform
import subprocess
import zipfile
from p3_depth.residual.records import sha, write_json, now


def load(root):
    return json.loads((root/'configs/encoder_residual.json').read_text(encoding='utf8'))


def reference_path(root, study, name, seed):
    if name == 'S0':
        return root/study['parent_output']/'runs'/f'frozen_features_seed{seed}'
    if name == 'SS1':
        return root/'outputs/part3_improvement/stylesketch_20261120/runs'/f'SS1_seed{seed}'
    return root/study['reference_output']/'followups'/f'factorial_{name}_seed{seed}'


def audit_references(root, study):
    split = json.loads((root/study['split']).read_text())
    parent = json.loads((root/study['parent_output']/'cache/development/metadata.json').read_text())
    frozen = json.loads((root/study['reference_output']/'residual_frozen.json').read_text())
    for name, digest in frozen['source_sha256'].items():
        assert sha(root/name) == digest, ('Reference source changed', name)
    old = json.loads((root/'configs/fusion_lora_study.json').read_text())
    assert study['training'] == old['training'] and study['protocol'] == old['protocol']
    records = []
    for candidate in study['queue']:
        name = study['candidates'][candidate]['reference']
        for seed in study['seeds']:
            path = reference_path(root, study, name, seed)
            cfg = json.loads((path/'config.json').read_text())
            result = json.loads((path/'result.json').read_text())
            assert cfg['fit_ids'] == split['fit_ids'] and cfg['validation_ids'] == split['validation_ids']
            assert cfg['training'] == study['training'] and cfg['cache_sha256'] == parent['cache_sha256']
            assert result['status'] == 'complete' and result['validation']['images'] == 218
            weights = result.get('checkpoint_sha256', {}) or {'best.pt': result['best_checkpoint_sha256']}
            for filename, digest in weights.items():
                assert sha(path/filename) == digest, path/filename
            records.append({'method':name,'seed':seed,'path':path.relative_to(root).as_posix(),
                'files':{f:sha(path/f) for f in ('config.json','result.json','history.json','validation/per_image.csv')}})
    out = root/study['output']
    record = {'status':'pass','at':now(),'records':records,'split_sha256':sha(root/study['split']),
              'cache_sha256':parent['cache_sha256'],'prior_source_sha256':frozen['source_sha256'],
              'training_and_protocol_equal':True,'old_test_results_not_imported':True}
    write_json(out/'reference_audit.json',record)
    return record


def source_paths(root):
    names = ['configs/encoder_residual.json','configs/edge_association_split.json',
             'docs/encoder_residual_plan.md','scripts/part3_improvement/run_encoder_residual.py',
             'src/p3_depth/models/runtime.py','src/p3_depth/data/nyuv2.py',
             'src/p3_depth/residual_inputs/model.py','src/p3_depth/residual_inputs/evaluation.py',
             'src/p3_depth/adaptation_study/model.py','src/p3_depth/stylesketch_residual/model.py',
             'src/p3_depth/stylesketch_residual/low_memory.py',
             'src/p3_depth/stylesketch_residual/data.py','src/p3_depth/stylesketch_residual/common.py']
    paths = [root/n for n in names]
    for folder in ('residual','evaluation','encoder_residual'):
        paths += sorted((root/'src/p3_depth'/folder).glob('*.py'))
    return sorted(set(paths))


def freeze(root, study, preflights):
    import torch
    out = root/study['output']
    assert not (out/'design_frozen.json').exists()
    hashes = {p.relative_to(root).as_posix():sha(p) for p in source_paths(root)}
    with zipfile.ZipFile(out/'numerical_source.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name in hashes:
            z.write(root/name,name)
    record = {'at':now(),'status':'fixed_before_formal_training','study':study,
        'source_sha256':hashes,'source_archive_sha256':sha(out/'numerical_source.zip'),
        'feature_metadata_sha256':sha(out/'features/metadata.json'),
        'reference_audit_sha256':sha(out/'reference_audit.json'),
        'preflight_sha256':sha(out/'preflight.json'),
        'parameters':{name:r['parameters'] for name,r in preflights.items()},
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,
        'gpu':torch.cuda.get_device_name(),'no_architecture_search':True}
    write_json(out/'design_frozen.json',record)
    return record


def check_freeze(root, study):
    out = root/study['output']; record = json.loads((out/'design_frozen.json').read_text())
    assert record['study'] == study
    for name,digest in record['source_sha256'].items():
        assert sha(root/name) == digest, ('Frozen source changed',name)
    for name,key in [('features/metadata.json','feature_metadata_sha256'),('numerical_source.zip','source_archive_sha256'),('preflight.json','preflight_sha256'),('reference_audit.json','reference_audit_sha256')]:
        assert sha(out/name) == record[key],name
    return record
