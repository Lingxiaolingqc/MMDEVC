"""Bind frozen fusion parameters to verified, registered development clips."""

import csv
import json
from pathlib import Path

from p3_depth.models.runtime import file_hash


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def local_path(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Experiment evidence must stay inside the repository')
    return path


def relative(root, path):
    return local_path(root, path).relative_to(root.resolve()).as_posix()


def registered_clips(root):
    with (root / 'manifests/videos.csv').open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    clips, groups, sources = {}, {}, set()
    for row in rows:
        clip_id, group, split = row['clip_id'], row['scene_group'], row['split']
        if not clip_id or not group or clip_id in clips or split not in ('dev', 'test'):
            raise ValueError('Registered clips need unique IDs, scene groups and dev/test splits')
        if group in groups and groups[group] != split:
            raise ValueError('A scene group cannot cross dev/test')
        groups[group] = split
        notes = dict(part.strip().split('=', 1) for part in row['notes'].split(';'))
        cache = local_path(root, notes['cache'])
        meta = read_json(cache / 'metadata.json')
        if (meta.get('self_recorded') is not True or meta.get('scope') != 'self_recorded_project'
                or meta.get('max_frames') is not None):
            raise ValueError('Only complete self-recorded clips may enter formal selection')
        if any(meta[key] != row[key] for key in ('clip_id', 'scene_group', 'split', 'scene_type')):
            raise ValueError('Cache identity differs from the registered manifest')
        source = notes['source_sha256']
        if source in sources or source != meta['source_sha256']:
            raise ValueError('Duplicate or changed source video')
        sources.add(source)
        raw = local_path(root, Path('data/videos') / row['video_path'])
        if file_hash(raw) != source:
            raise ValueError('Registered original video changed')
        for name, key in [('frames.npy', 'frames_sha256'), ('frames.csv', 'index_sha256')]:
            if file_hash(cache / name) != meta[key]:
                raise ValueError('Registered RGB/time cache changed')
        clips[clip_id] = {'row': row, 'cache': cache, 'metadata': meta}
    if not clips:
        raise ValueError('Register self-recorded development and test clips first')
    return clips


def suite_artifacts():
    paths = ['comparison.csv']
    for variant in ('E0', 'E1', 'E2', 'E3', 'E4', 'E5'):
        paths += [f'{variant}/metrics.json', f'{variant}/per_pair.csv']
        if variant != 'E5':
            paths += [f'{variant}/{name}' for name in
                      ('metadata.json', 'depths.npy', 'valid.npy', 'history_weight.npy')]
    return paths


def checked_suite(root, suite, clips):
    suite = local_path(root, suite)
    meta = read_json(suite / 'metadata.json')
    clip_id = meta['clip_id']
    if (meta['status'] != 'complete' or meta['split'] != 'dev'
            or meta['scope'] != 'self_recorded_project' or clip_id not in clips
            or clips[clip_id]['row']['split'] != 'dev'):
        raise ValueError('Candidate suites must use registered self-recorded dev clips')
    if file_hash(clips[clip_id]['cache'] / 'metadata.json') != meta['cache_metadata_sha256']:
        raise ValueError('Candidate uses a different registered cache')
    verification = read_json(suite / 'verification.json')
    expected_inputs = [meta[k] for k in ('cache_metadata_sha256', 'base_metadata_sha256',
                                        'vda_metadata_sha256', 'flow_metadata_sha256')]
    if (verification.get('schema_version') != 1 or verification.get('status') != 'verified'
            or verification['suite_metadata_sha256'] != file_hash(suite / 'metadata.json')
            or set(verification['artifacts_sha256']) != set(suite_artifacts())
            or sorted(verification['input_metadata_sha256'].values()) != sorted(expected_inputs)):
        raise ValueError('Run verify_fusion_suite.py on the completed candidate first')
    for name, expected in verification['artifacts_sha256'].items():
        if file_hash(suite / name) != expected:
            raise ValueError('Verified candidate artifact changed')
    for name, expected in verification['input_metadata_sha256'].items():
        if file_hash(local_path(root, name)) != expected:
            raise ValueError('Verified candidate input metadata changed')
    if file_hash(suite / 'comparison.csv') != meta['comparison_sha256']:
        raise ValueError('Candidate comparison changed')
    with (suite / 'comparison.csv').open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if [r['variant'] for r in rows] != ['E0', 'E1', 'E2', 'E3', 'E4', 'E5']:
        raise ValueError('Candidate needs all six ablations')
    counts = {int(r['reference_pixels']) for r in rows}
    if len(counts) != 1 or next(iter(counts)) <= 0:
        raise ValueError('Candidates need a nonempty common evaluation region')
    record = {'clip_id': clip_id, 'suite': relative(root, suite),
              'metadata_sha256': file_hash(suite / 'metadata.json'),
              'verification_sha256': file_hash(suite / 'verification.json'),
              'comparison_sha256': meta['comparison_sha256']}
    return meta, record, rows


def selection_evidence(root, candidates, selected, review_path):
    """No automatic winner: the review must address stability and motion response."""
    clips = registered_clips(root)
    dev_ids = {key for key, value in clips.items() if value['row']['split'] == 'dev'}
    if not dev_ids or not any(c['row']['split'] == 'test' for c in clips.values()):
        raise ValueError('Register both development and independent test clips before freezing')
    if selected not in candidates:
        raise ValueError('Selected candidate was not evaluated')
    review_path = local_path(root, review_path)
    review = read_json(review_path)
    if (review.get('status') != 'complete' or review.get('selected_candidate') != selected
            or review.get('reviewer_kind') not in ('assistant', 'member')
            or not str(review.get('rationale', '')).strip()
            or set(review.get('candidate_observations', {})) != set(candidates)):
        raise ValueError('Complete the candidate review with an explicit selection and rationale')
    records, tables, sources_by_clip, parameters = {}, [], {}, {}
    for name, suites in sorted(candidates.items()):
        observations = review['candidate_observations'][name]
        if set(observations) != dev_ids:
            raise ValueError('Review every registered development clip for every candidate')
        for observation in observations.values():
            for key in ('stability', 'motion_response', 'occlusion_and_edges'):
                if not isinstance(observation.get(key), str) or not observation[key].strip():
                    raise ValueError('Record stability, motion response and occlusion/edge observations')
        seen, records[name] = set(), []
        for suite in sorted(suites):
            meta, record, rows = checked_suite(root, suite, clips)
            clip_id = record['clip_id']
            if clip_id in seen:
                raise ValueError('Duplicate clip within a candidate')
            seen.add(clip_id)
            p = meta['config']['parameters']
            if name in parameters and parameters[name] != p:
                raise ValueError('A candidate must use the same parameters on every development clip')
            parameters[name] = p
            source = {k: meta[k] for k in ('cache_metadata_sha256', 'base_metadata_sha256',
                                          'vda_metadata_sha256', 'flow_metadata_sha256', 'implementation_sha256')}
            if clip_id in sources_by_clip and sources_by_clip[clip_id] != source:
                raise ValueError('Candidates must share model predictions, flow, evaluation and implementation')
            sources_by_clip[clip_id] = source
            records[name].append(record)
            tables += [{'candidate': name, 'clip_id': clip_id, **row} for row in rows]
        if seen != dev_ids:
            raise ValueError('Every candidate must cover all registered development clips')
    proof = {'schema_version': 1, 'split': 'dev', 'dev_clip_ids': sorted(dev_ids),
             'manifest_sha256': file_hash(root / 'manifests/videos.csv'),
             'selected_candidate': selected, 'selected_parameters': parameters[selected],
             'candidates': records, 'review_path': relative(root, review_path),
             'review_sha256': file_hash(review_path), 'reviewer_kind': review['reviewer_kind']}
    return proof, tables


def validate_test_freeze(root, config, cache_metadata):
    evidence = config.get('selection_evidence')
    if config.get('status') != 'frozen' or not isinstance(evidence, dict):
        raise ValueError('Test evaluation requires parameters frozen from development evidence')
    path = local_path(root, evidence['path'])
    if file_hash(path) != evidence['sha256']:
        raise ValueError('Development selection evidence hash mismatch')
    proof = read_json(path)
    if proof.get('schema_version') != 1 or proof['manifest_sha256'] != file_hash(root / 'manifests/videos.csv'):
        raise ValueError('Registered dev/test split changed after parameter selection')
    candidates = {name: [r['suite'] for r in records] for name, records in proof['candidates'].items()}
    current, _ = selection_evidence(root, candidates, proof['selected_candidate'], proof['review_path'])
    if current != proof or proof['selected_parameters'] != config['parameters']:
        raise ValueError('Frozen parameters or development evidence changed')
    clips = registered_clips(root)
    clip = clips.get(cache_metadata['clip_id'])
    if clip is None or clip['row']['split'] != 'test' or clip['metadata'] != cache_metadata:
        raise ValueError('Test input must be the independently registered complete test cache')
