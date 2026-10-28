"""Download pinned official Small checkpoints; validate LFS size and SHA256."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['all', 'dav2_relative', 'dav2_metric', 'vda_metric'], default='all')
    parser.add_argument('--checkpoint-root', type=Path, default=ROOT / 'checkpoints')
    args = parser.parse_args()
    assets = json.loads((ROOT / 'environments/model_assets.json').read_text(encoding='utf-8'))
    for name, asset in assets.items():
        if args.model not in ('all', name):
            continue
        path = args.checkpoint_root / name / asset['filename']
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://huggingface.co/{asset['repository']}/resolve/{asset['revision']}/{asset['filename']}"
            partial = path.with_suffix(path.suffix + '.partial')
            print(f'Downloading {name}: {asset["bytes"]} bytes', flush=True)
            with urlopen(url, timeout=90) as response, partial.open('wb') as f:
                for block in iter(lambda: response.read(1024 * 1024), b''):
                    f.write(block)
            if partial.stat().st_size != asset['bytes'] or sha256(partial) != asset['sha256']:
                raise ValueError(f'Incomplete or incorrect checkpoint: {partial}')
            partial.replace(path)
        if path.stat().st_size != asset['bytes'] or sha256(path) != asset['sha256']:
            raise ValueError(f'Existing checkpoint differs from the pinned release: {path}')
        print(f'PASS {name}: pinned size and SHA256', flush=True)


if __name__ == '__main__':
    main()
