"""Archive outputs and tracked reproduction context; verify every entry, never delete."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(path):
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read('ARCHIVE_MANIFEST.json'))
        expected = {r['path'] for r in manifest['files']} | {'ARCHIVE_MANIFEST.json'}
        if set(z.namelist()) != expected or len(z.namelist()) != len(expected):
            raise ValueError('Archive member list differs')
        for row in manifest['files']:
            h = hashlib.sha256(); size = 0
            with z.open(row['path']) as f:
                for block in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(block); size += len(block)
            if h.hexdigest() != row['sha256'] or size != row['bytes']:
                raise ValueError('Archive verification failed: ' + row['path'])
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    output = (ROOT / a.output).resolve()
    if output.is_relative_to(ROOT / 'outputs') or output.suffix != '.zip':
        raise ValueError('Keep the archive outside outputs and use .zip')
    if a.check_only:
        m = verify(output)
        print(f'PASS: {len(m["files"])} archived files verified'); return
    if output.exists():
        raise FileExistsError('Preserve the existing archive')
    tracked = subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True).splitlines()
    context = [ROOT / name for name in tracked if name.startswith(
        ('scripts/', 'src/', 'configs/', 'manifests/', 'report/', 'docs/', 'environments/', 'results/', 'tests/'))
        or name in ('README.md', 'PARTS.md', 'pyproject.toml', '.gitignore', '.gitattributes')]
    files = sorted({*filter(Path.is_file, context), *filter(Path.is_file, (ROOT/'outputs').rglob('*'))})
    manifest = {'schema_version': 1, 'purpose': 'pre-reorganization backup; no deletion authorized by this tool',
                'git_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'git_status': subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True), 'files': []}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for index, source in enumerate(files):
            if not source.resolve().is_relative_to(ROOT):
                raise ValueError('Source escaped repository: ' + str(source))
            name = source.relative_to(ROOT).as_posix(); h = hashlib.sha256(); size = 0
            with source.open('rb') as f, z.open(name, 'w', force_zip64=True) as dest:
                for block in iter(lambda: f.read(1024 * 1024), b''):
                    dest.write(block); h.update(block); size += len(block)
            manifest['files'].append({'path': name, 'bytes': size, 'sha256': h.hexdigest()})
            if index % 1000 == 0:
                print(f'Archived {index+1}/{len(files)} files', flush=True)
        z.writestr('ARCHIVE_MANIFEST.json', json.dumps(manifest, ensure_ascii=False, indent=2))
    verify(output)
    # Detect a source changing while the archive was being written.
    for row in manifest['files']:
        if digest(ROOT / row['path']) != row['sha256']:
            raise ValueError('Source changed while archiving: ' + row['path'])
    receipt = {'status': 'verified', 'archive': str(output), 'sha256': digest(output),
               'files': len(files), 'source_bytes': sum(r['bytes'] for r in manifest['files']),
               'archive_bytes': output.stat().st_size, 'deleted_files': 0}
    output.with_suffix('.verification.json').write_text(json.dumps(receipt, indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
