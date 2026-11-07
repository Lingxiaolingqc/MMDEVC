"""Resolve one shared layout for new runs; explicit output paths still take priority."""
import re
from _common import ROOT, read_yaml, repo_path

CURRENT_VIDEO_MANIFEST = 'manifests/videos_current.csv'


def output_path(key, paths_config='configs/paths.local.yaml', **fields):
    layout = read_yaml(ROOT / 'configs/output_layout.yaml')
    settings_path = repo_path(paths_config)
    settings = read_yaml(settings_path) if settings_path.exists() else {}
    base = repo_path(settings.get('output_root', 'outputs'))
    for name, value in fields.items():
        if not re.fullmatch(r'[A-Za-z0-9_-]+', str(value)):
            raise ValueError(f'Unsafe output identifier: {name}={value}')
    relative = layout['paths'][key].format(**fields)
    destination = (base / relative).resolve()
    if not destination.is_relative_to(base) or destination == base:
        raise ValueError('Output layout escaped its root')
    return destination


def choose_output(explicit, key, paths_config='configs/paths.local.yaml', **fields):
    return repo_path(explicit) if explicit else output_path(key, paths_config, **fields)


def clip_output(key, cache_metadata, explicit=None, **fields):
    return choose_output(explicit, key, clip=cache_metadata['clip_id'],
                         split=cache_metadata['split'], **fields)
