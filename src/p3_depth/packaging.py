"""Collect verified video exports and audit every archived byte, without raw code/data."""

import csv
import hashlib
import json
import math
from pathlib import Path
import re
import zipfile

import yaml
from p3_depth.models.runtime import file_hash
from p3_depth.temporal.selection import local_path, read_json, registered_clips, suite_artifacts, validate_test_freeze


def collect_exports(root, plan):
    scope = plan.get('scope')
    formal = scope == 'self_recorded_project'
    if plan.get('schema_version') != 1 or scope not in ('self_recorded_project', 'pipeline_check_not_final_evaluation'):
        raise ValueError('Declare schema_version=1 and the actual export scope')
    clips = plan.get('clips', [])
    ids = [clip['clip_id'] for clip in clips]
    if not ids or len(set(ids)) != len(ids) or any(not re.fullmatch(r'[A-Za-z0-9_-]+', x) for x in ids):
        raise ValueError('Provide unique nonempty clip IDs')
    registry = registered_clips(root) if formal else {}
    if formal and set(ids) != set(registry):
        raise ValueError('Formal archive must cover every registered development and test clip')
    files, clip_summary, demos = {}, [], []

    def add(path, archive_name, expected=None):
        path = local_path(root, path)
        if not path.is_file() or archive_name in files:
            raise ValueError('Missing source or duplicate archive destination')
        digest = file_hash(path)
        if expected is not None and digest != expected:
            raise ValueError(f'Export hash changed: {archive_name}')
        files[archive_name] = {'path': path, 'sha256': digest, 'bytes': path.stat().st_size}

    config = None
    if formal:
        config_path = local_path(root, plan['frozen_config'])
        config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        test = next((item['metadata'] for item in registry.values() if item['row']['split'] == 'test'), None)
        if test is None:
            raise ValueError('Formal archive needs independent test clips')
        validate_test_freeze(root, config, test)
        add(config_path, 'evidence/fusion_frozen.yaml')
        proof_path = local_path(root, config['selection_evidence']['path'])
        proof = read_json(proof_path)
        add(proof_path, 'evidence/selection.json', config['selection_evidence']['sha256'])
        add(proof['review_path'], 'evidence/development_review.json', proof['review_sha256'])
        add(root/'manifests/videos.csv', 'evidence/videos.csv', proof['manifest_sha256'])

    for clip in clips:
        clip_id = clip['clip_id']
        cache = local_path(root, clip['cache'])
        cm = read_json(cache/'metadata.json')
        if cm['clip_id'] != clip_id or cm['scope'] != scope:
            raise ValueError('Export scope/clip ID differs from the cache')
        if formal and cm != registry[clip_id]['metadata']:
            raise ValueError('Formal export does not use the registered complete cache')
        add(cache/'metadata.json', f'clips/{clip_id}/cache_metadata.json')
        add(cache/'frames.csv', f'clips/{clip_id}/timestamps.csv', cm['index_sha256'])
        seen, scales, expected_methods = {}, set(), {}
        if formal:
            suite = local_path(root, clip['suite'])
            sm, receipt = read_json(suite/'metadata.json'), read_json(suite/'verification.json')
            if (sm['status'] != 'complete' or sm['scope'] != scope or sm['clip_id'] != clip_id
                    or sm['cache_metadata_sha256'] != file_hash(cache/'metadata.json')
                    or sm['config']['parameters'] != config['parameters']
                    or (cm['split'] == 'test' and sm['config'] != config)
                    or receipt['status'] != 'verified'
                    or receipt['suite_metadata_sha256'] != file_hash(suite/'metadata.json')
                    or set(receipt['artifacts_sha256']) != set(suite_artifacts())):
                raise ValueError('Formal exports need verified suites at the selected parameters')
            for name, digest in receipt['artifacts_sha256'].items():
                if file_hash(suite/name) != digest:
                    raise ValueError('Verified suite artifact changed')
                if Path(name).suffix in ('.csv', '.json'):
                    add(suite/name, f'clips/{clip_id}/metrics/{name}', digest)
            add(suite/'metadata.json', f'clips/{clip_id}/metrics/suite_metadata.json')
            add(suite/'verification.json', f'clips/{clip_id}/metrics/verification.json')
            expected_methods = {f'E{i}': file_hash(suite/f'E{i}'/'metadata.json') for i in range(5)}
            expected_methods['vda_metric'] = file_hash(local_path(root, sm['E5_predictions'])/'metadata.json')
        for label, folder in clip['renders'].items():
            if not re.fullmatch(r'[A-Za-z0-9_-]+', label):
                raise ValueError('Use simple render labels for archive folders')
            folder = local_path(root, folder)
            meta = read_json(folder/'metadata.json')
            if (meta['status'] != 'complete' or meta['scope'] != scope or meta['frames'] != cm['frames']
                    or meta['cache_frames_sha256'] != cm['frames_sha256']
                    or meta['cache_index_sha256'] != cm['index_sha256'] or meta['raw_predictions_modified'] is not False):
                raise ValueError('Rendered comparison must preserve the selected RGB/time cache and predictions')
            for item in meta['inputs']:
                name = item['model']
                source = local_path(root, item['run'])/'metadata.json'
                if file_hash(source) != item['metadata_sha256']:
                    raise ValueError('Rendered model provenance changed')
                if name in seen and seen[name] != item['metadata_sha256']:
                    raise ValueError('One archive clip cannot mix different runs of the same method')
                if name not in seen:
                    if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
                        raise ValueError('Invalid model name')
                    add(source, f'clips/{clip_id}/models/{name}.json', item['metadata_sha256'])
                seen[name] = item['metadata_sha256']
                if item['unit'] == 'm':
                    scales.add((item['low'], item['high']))
                if name in expected_methods and item['metadata_sha256'] != expected_methods[name]:
                    raise ValueError('Rendered prediction differs from the selected verified suite')
            if len(scales) > 1:
                raise ValueError('Metric methods on one clip must share a fixed display range')
            for name, key in [('comparison.mp4', 'video_sha256'), ('preview.png', 'preview_sha256')]:
                add(folder/name, f'clips/{clip_id}/{label}/{name}', meta[key])
            add(folder/'metadata.json', f'clips/{clip_id}/{label}/metadata.json')
        if not seen or (formal and not ({'dav2_relative', 'E0', 'E1', 'E2', 'E3', 'E4', 'vda_metric'} <= set(seen))):
            raise ValueError('Formal videos must include relative baseline and all E0-E5 methods')
        if formal or clip.get('relative_diagnostics'):
            folder = local_path(root, clip['relative_diagnostics'])
            meta = read_json(folder/'metadata.json')
            if (meta['status'] != 'complete' or meta['scope'] != scope or meta['clip_id'] != clip_id
                    or meta['source_metadata_sha256']['cache'] != file_hash(cache/'metadata.json')
                    or meta['source_metadata_sha256']['run'] != seen.get('dav2_relative')):
                raise ValueError('Relative diagnostics must match the rendered relative prediction')
            for name in ('metrics.json', 'per_pair.csv', 'per_frame.csv', 'diagnostics.png'):
                add(folder/name, f'clips/{clip_id}/relative_diagnostics/{name}', meta['artifacts_sha256'][name])
            add(folder/'metadata.json', f'clips/{clip_id}/relative_diagnostics/metadata.json')
        if clip.get('demo'):
            folder = local_path(root, clip['demo'])
            meta = read_json(folder/'metadata.json')
            if (meta['status'] != 'complete' or meta['scope'] != scope or meta['clip_id'] != clip_id
                    or meta['cache_metadata_sha256'] != file_hash(cache/'metadata.json')
                    or not meta['verification']['decoded_timestamps_match']
                    or not meta['verification']['trace_matches_raw_predictions']
                    or any(seen.get(m['name']) != m['metadata_sha256'] for m in meta['methods'])):
                raise ValueError('Application demo must match the exported methods and verified timestamps/values')
            if formal and (cm['split'] != 'test' or any(m['provisional'] for m in meta['methods'])):
                raise ValueError('Formal application demo must use frozen independent test results')
            for name in ('demo.mp4', 'preview.png', 'point_trace.csv'):
                add(folder/name, f'application/{clip_id}/{name}', meta['files'][name])
            add(folder/'metadata.json', f'application/{clip_id}/metadata.json')
            demos.append(clip_id)
        clip_summary.append({'clip_id': clip_id, 'split': cm['split'], 'scene_type': cm['scene_type'],
                             'frames': cm['frames'], 'methods': sorted(seen)})
    if not demos:
        raise ValueError('A depth-results archive must include a verified RGB/depth/application demonstration')
    if formal:
        cases = local_path(root, plan['failure_cases'])
        with cases.open(encoding='utf-8', newline='') as handle:
            rows = list(csv.DictReader(handle))
        if not {'mirror', 'glass', 'sky', 'motion_blur', 'thin_object'} <= {r['category'] for r in rows}:
            raise ValueError('Index all five required qualitative conditions before final packaging')
        for row in rows:
            start, end = float(row['start_s']), float(row['end_s'])
            if (row['clip_id'] not in registry or not math.isfinite(start+end) or not 0 <= start < end
                    or end > registry[row['clip_id']]['metadata']['sampled_duration_s']
                    or not row['observation'].strip()):
                raise ValueError('Failure/condition records need real clip intervals and observations')
        add(cases, 'evidence/failure_cases.csv')
    return files, {'schema_version': 1, 'scope': scope, 'clips': clip_summary, 'application_clips': demos}


def write_archive(files, manifest, output):
    output = Path(output)
    if output.exists() or output.with_suffix('.verification.json').exists():
        raise FileExistsError('Use a new archive output path')
    formal = manifest['scope'] == 'self_recorded_project'
    if not formal and output.name == 'depth_results.zip':
        raise ValueError('Use a clearly labelled smoke archive name for example footage')
    manifest = {**manifest, 'files': {name: {'sha256': item['sha256'], 'bytes': item['bytes']}
                                    for name, item in sorted(files.items())}}
    readme = ('MMDEVC depth video outputs\n\nScope: ' + manifest['scope'] + '\n' +
              ('Prepared video archive. The report and course submission are separate.\n' if formal else
               'PIPELINE CHECK ONLY. Example footage; not a formal course submission.\n') +
              '\nclips/: RGB and fixed-scale depth comparisons, timestamps and provenance.\n'
              'application/: RGB/depth/point-Z demonstration and raw-value trace.\n'
              'manifest.json: every source payload, its size and SHA256.\n'
              'Relative values are arbitrary inverse-depth-like units. Metric Z is camera-axis depth.\n'
              'Raw arrays, weights, datasets, source code and personal acceptance notes remain outside this archive.\n')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.zip.partial')
    if temporary.exists():
        raise FileExistsError('A partial archive already exists; inspect it before retrying')
    try:
        with zipfile.ZipFile(temporary, 'x', allowZip64=True) as archive:
            archive.writestr('README.txt', readme)
            archive.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
            for name, item in sorted(files.items()):
                archive.write(item['path'], name, compress_type=zipfile.ZIP_STORED if Path(name).suffix in ('.mp4', '.png') else zipfile.ZIP_DEFLATED)
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None or len(archive.namelist()) != len(files)+2:
                raise ValueError('Archive CRC or file count verification failed')
            for name, item in files.items():
                digest = hashlib.sha256()
                with archive.open(name) as handle:
                    for block in iter(lambda: handle.read(1024*1024), b''):
                        digest.update(block)
                if digest.hexdigest() != item['sha256']:
                    raise ValueError('Archived payload differs from verified source')
        temporary.rename(output)
    except Exception:
        if temporary.is_file():
            temporary.unlink()
        raise
    record = {'status': 'verified', 'scope': manifest['scope'], 'archive_sha256': file_hash(output),
              'bytes': output.stat().st_size, 'payload_files': len(files), 'crc_checked': True,
              'all_payload_sha256_match': True, 'course_submission_performed': False}
    output.with_suffix('.verification.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    return record
