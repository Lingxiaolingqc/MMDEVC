"""Independent experiment identity; never changes prior freezes."""
import json
import platform
import subprocess
import zipfile
from p3_depth.residual.records import sha, write_json, now, identity


def load(root):
    return json.loads((root/'configs/stylesketch_residual.json').read_text(encoding='utf8'))


def source_paths(root):
    names = ['configs/stylesketch_residual.json', 'configs/edge_association_split.json',
             'docs/stylesketch_residual_plan.md', 'scripts/part3_improvement/run_stylesketch_residual.py',
             'src/p3_depth/models/runtime.py', 'src/p3_depth/data/nyuv2.py',
             'src/p3_depth/residual/model.py', 'src/p3_depth/residual/job_schedule.py',
             'src/p3_depth/residual/evaluation.py', 'src/p3_depth/residual/records.py',
             'src/p3_depth/residual/training.py', 'src/p3_depth/residual_inputs/data.py',
             'src/p3_depth/residual_inputs/evaluation.py',
             'src/p3_depth/evaluation/course.py', 'src/p3_depth/evaluation/boundary.py',
             'src/p3_depth/evaluation/depth.py', 'src/p3_depth/evaluation/rgb_depth_edges.py']
    return [root/n for n in names] + sorted((root/'src/p3_depth/stylesketch_residual').glob('*.py'))


def audit_references(root, study):
    prior = root/study['reference_output']
    frozen = json.loads((prior/'residual_frozen.json').read_text())
    for name, digest in frozen['source_sha256'].items():
        assert sha(root/name) == digest, ('Prior numerical source changed', name)
    split = json.loads((root/study['split']).read_text())
    parent = json.loads((root/study['parent_output']/'cache/development/metadata.json').read_text())
    original_study = json.loads((root/'configs/fusion_lora_study.json').read_text())
    assert original_study['training'] == study['training'] and original_study['protocol'] == study['protocol']
    records = []
    for name in ('F3', 'P3', 'G3'):
        for seed in study['seeds']:
            run = prior/'followups'/f'factorial_{name}_seed{seed}'
            cfg = json.loads((run/'config.json').read_text())
            result = json.loads((run/'result.json').read_text())
            assert cfg['fit_ids'] == split['fit_ids'] and cfg['validation_ids'] == split['validation_ids']
            assert cfg['training'] == study['training'] and cfg['cache_sha256'] == parent['cache_sha256']
            assert result['status'] == 'complete' and result['validation']['images'] == 218
            assert all(sha(run/f) == v for f, v in result['checkpoint_sha256'].items())
            records.append({'method': name, 'seed': seed, 'path': run.relative_to(root).as_posix(),
                            'sha256': {f: sha(run/f) for f in ('config.json', 'result.json', 'validation/per_image.csv', 'history.json')}})
    b0 = prior/'reference_B0'
    assert (b0/'metrics.json').exists()
    result = {'at': now(), 'status': 'pass', 'reuse': records, 'prior_freeze_sha256': sha(prior/'residual_frozen.json'),
              'split_sha256': sha(root/study['split']), 'cache_sha256': parent['cache_sha256'],
              'shared_numerical_sources_verified': frozen['source_sha256'],
              'B0': {f: sha(b0/f) for f in ('metrics.json', 'per_image.csv')},
              'protocol_equal': True, 'training_equal': True}
    write_json(root/study['output']/'reference_audit.json', result)
    return result


def freeze(root, study, preflight):
    import torch
    out = root/study['output']
    hashes = {p.relative_to(root).as_posix(): sha(p) for p in source_paths(root)}
    official = out/'official/source.json'
    record = {'at': now(), 'status': 'fixed_before_formal_training', 'study': study,
              'config_sha256': sha(root/'configs/stylesketch_residual.json'), 'source_sha256': hashes,
              'official_source_sha256': sha(official), 'preflight_sha256': sha(out/'preflight.json'),
              'feature_metadata_sha256': sha(out/'features/metadata.json'),
              'reference_audit_sha256': sha(out/'reference_audit.json'),
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
              'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
              'gpu': torch.cuda.get_device_name(), 'parameters': preflight['parameters'],
              'architecture_search': False, 'three_seeds_only': True}
    path = out/'design_frozen.json'
    assert not path.exists(), 'Do not overwrite pretraining freeze'
    with zipfile.ZipFile(out/'numerical_source.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for name in hashes:
            archive.write(root/name, name)
        for path2 in (out/'official').glob('*'):
            archive.write(path2, 'official/'+path2.name)
    record['source_archive_sha256'] = sha(out/'numerical_source.zip')
    write_json(path, record)
    return record


def check_freeze(root, study):
    out = root/study['output']
    record = json.loads((out/'design_frozen.json').read_text())
    assert record['study'] == study
    for name, value in record['source_sha256'].items():
        assert sha(root/name) == value, ('Frozen source changed', name)
    assert sha(out/'features/metadata.json') == record['feature_metadata_sha256']
    assert sha(out/'numerical_source.zip') == record['source_archive_sha256']
    return record
