"""Atomic experiment records and explicitly scoped retention."""
from datetime import datetime, timezone
import hashlib
import json
import platform
import os
from pathlib import Path
import subprocess
from importlib.metadata import version


def native_path(path):
    """Use Windows extended paths for nested immutable source snapshots."""
    value = Path(path).absolute()
    name = str(value)
    if os.name == 'nt' and not name.startswith('\\\\?\\'):
        return Path('\\\\?\\UNC\\' + name[2:] if name.startswith('\\\\') else '\\\\?\\' + name)
    return value


def sha(path):
    digest = hashlib.sha256()
    with native_path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = native_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def append_json(path, value):
    with native_path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'recorded_at': now(), **value}, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()


def source_files(root):
    names = ['configs/residual_study.json', 'configs/edge_association_split.json',
             'docs/residual_research_plan.md', 'src/p3_depth/data/nyuv2.py',
             'src/p3_depth/models/runtime.py', 'src/p3_depth/evaluation/course.py',
             'src/p3_depth/evaluation/depth.py', 'src/p3_depth/evaluation/boundary.py',
             'src/p3_depth/evaluation/rgb_depth_edges.py',
             'src/p3_depth/temporal/fusion.py']
    optional=[root/'configs'/name for name in ('residual_confirmation.json',
               'residual_combinations.json', 'residual_followup_selection.json')]
    return [root / x for x in names] + [path for path in optional if path.exists()] + sorted((root / 'src/p3_depth/residual').glob('*.py')) + sorted((root / 'scripts/part3_improvement').glob('*residual*.py'))


def snapshot(root, output):
    import torch
    import numpy
    import cv2
    hashes = {}
    for path in source_files(root):
        name = path.relative_to(root).as_posix()
        target = native_path(output / 'source_at_run' / name)
        target.parent.mkdir(parents=True, exist_ok=True)
        # CopyFile2 used by shutil.copy2 can reject deep Windows/OneDrive paths.
        # These small text sources need exact contents, not filesystem metadata.
        target.write_bytes(native_path(path).read_bytes())
        hashes[name] = sha(path)
    provenance = {'created_at': now(), 'source_sha256': hashes,
                  'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                  'git_status': subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip(),
                  'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
                  'numpy': numpy.__version__, 'opencv': cv2.__version__,
                  'packages': {name:version(name) for name in ('torch','numpy','scipy','h5py','opencv-python','matplotlib','Pillow','PyYAML','safetensors')},
                  'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None}
    write_json(output / 'provenance.json', provenance)
    return provenance


def remove_completed_resume(study_root, run):
    """Never recursive: remove only this completed run's replaceable resume file."""
    study_root, run = Path(study_root).resolve(), Path(run).resolve()
    if not run.is_relative_to(study_root) or run == study_root:
        raise ValueError('Retention path outside study')
    result = json.loads((run / 'result.json').read_text(encoding='utf-8'))
    if result['status'] != 'complete' or sha(run / 'best.pt') != result['best_checkpoint_sha256']:
        raise ValueError('Best checkpoint not verified; retain resume')
    final_hash=result.get('final_checkpoint_sha256') or result.get('checkpoint_sha256',{}).get('final.pt')
    if final_hash and sha(run/'final.pt')!=final_hash:
        raise ValueError('Final checkpoint not verified; retain resume')
    target = run / 'resume.pt'
    if not target.exists():
        return
    record = {'path': str(target), 'bytes': target.stat().st_size, 'sha256': sha(target),
              'reason': 'Completed run; best checkpoint, full history and provenance retained',
              'reconstruction': 'Re-run saved config/source with recorded seed; best checkpoint remains available'}
    append_json(study_root / 'cleanup.jsonl', {**record, 'status': 'planned'})
    target.unlink()
    append_json(study_root / 'cleanup.jsonl', {**record, 'status': 'removed'})
