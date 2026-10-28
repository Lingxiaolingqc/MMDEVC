"""Prevent split leakage and parameter/evidence changes after development selection."""

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from p3_depth.models.runtime import file_hash
from p3_depth.temporal.selection import selection_evidence, suite_artifacts, validate_test_freeze


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True), encoding='utf-8')


class SelectionTests(unittest.TestCase):
    def setUp(self):
        base = ROOT / 'outputs/test-temp'
        base.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rows, self.metas = [], {}
        for clip, split in [('dev_static', 'dev'), ('dev_motion', 'dev'), ('test_static', 'test')]:
            cache = self.root / 'cache' / clip
            cache.mkdir(parents=True)
            raw = self.root / 'data/videos/raw' / (clip + '.mp4')
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_bytes(clip.encode())
            (cache / 'frames.npy').write_bytes(b'selection-test fixture, not decoded footage')
            (cache / 'frames.csv').write_text('timestamp_s\n0\n0.1\n')
            meta = {'clip_id': clip, 'split': split, 'scene_group': clip,
                    'scene_type': 'static' if 'static' in clip else 'lateral',
                    'self_recorded': True, 'scope': 'self_recorded_project', 'max_frames': None,
                    'source_sha256': file_hash(raw), 'frames_sha256': file_hash(cache / 'frames.npy'),
                    'index_sha256': file_hash(cache / 'frames.csv')}
            self.metas[clip] = meta
            write_json(cache / 'metadata.json', meta)
            self.rows.append({**{k: meta[k] for k in ('clip_id', 'split', 'scene_group', 'scene_type')},
                              'video_path': 'raw/' + raw.name,
                              'notes': f'source_sha256={file_hash(raw)}; cache=cache/{clip}'})
            for kind in ('baseline', 'vda', 'flow'):
                write_json(self.root / 'inputs' / clip / kind / 'metadata.json', {'kind': kind, 'clip_id': clip})
        self.write_manifest()
        self.candidates = {}
        for name, alpha in [('mild', .5), ('strong', .7)]:
            self.candidates[name] = []
            for clip in ('dev_static', 'dev_motion'):
                suite = self.root / 'suites' / name / clip
                self.candidates[name].append(suite.relative_to(self.root).as_posix())
                for artifact in suite_artifacts():
                    path = suite / artifact
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text('{}' if path.suffix == '.json' else 'fixture')
                (suite / 'comparison.csv').write_text('variant,reference_pixels\n' + ''.join(
                    f'E{i},10\n' for i in range(6)))
                write_json(suite / 'metadata.json', {
                    'status': 'complete', 'split': 'dev', 'scope': 'self_recorded_project', 'clip_id': clip,
                    'config': {'parameters': {'history_weight': alpha}},
                    'cache_metadata_sha256': file_hash(self.root / 'cache' / clip / 'metadata.json'),
                    'base_metadata_sha256': file_hash(self.root / 'inputs' / clip / 'baseline/metadata.json'),
                    'vda_metadata_sha256': file_hash(self.root / 'inputs' / clip / 'vda/metadata.json'),
                    'flow_metadata_sha256': file_hash(self.root / 'inputs' / clip / 'flow/metadata.json'),
                    'implementation_sha256': {'source.py': 'fixed'},
                    'comparison_sha256': file_hash(suite / 'comparison.csv')})
                self.receipt(suite)
        self.review = self.root / 'review.json'
        write_json(self.review, {'status': 'complete', 'reviewer_kind': 'assistant',
                                'selected_candidate': 'mild', 'rationale': 'Fixture selection tests only.',
                                'candidate_observations': {name: {clip: {
                                    'stability': 'Fixture observation', 'motion_response': 'Fixture observation',
                                    'occlusion_and_edges': 'Fixture observation'}
                                    for clip in ('dev_static', 'dev_motion')} for name in self.candidates}})

    def write_manifest(self):
        path = self.root / 'manifests/videos.csv'
        path.parent.mkdir(exist_ok=True)
        with path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def receipt(self, suite):
        clip = suite.name
        inputs = [self.root / 'cache' / clip / 'metadata.json'] + [
            self.root / 'inputs' / clip / kind / 'metadata.json' for kind in ('baseline', 'vda', 'flow')]
        write_json(suite / 'verification.json', {
            'schema_version': 1, 'status': 'verified',
            'suite_metadata_sha256': file_hash(suite / 'metadata.json'),
            'artifacts_sha256': {name: file_hash(suite / name) for name in suite_artifacts()},
            'input_metadata_sha256': {p.relative_to(self.root).as_posix(): file_hash(p) for p in inputs}})

    def freeze(self):
        proof, rows = selection_evidence(self.root, self.candidates, 'mild', self.review)
        path = self.root / 'selection.json'
        write_json(path, proof)
        config = {'status': 'frozen', 'parameters': proof['selected_parameters'],
                  'selection_evidence': {'path': 'selection.json', 'sha256': file_hash(path)}}
        return config, rows

    def test_verified_development_selection_allows_independent_test(self):
        config, rows = self.freeze()
        self.assertEqual(len(rows), 2 * 2 * 6)
        self.assertEqual(config['parameters']['history_weight'], .5)
        validate_test_freeze(self.root, config, self.metas['test_static'])
        with self.assertRaisesRegex(ValueError, 'independently registered'):
            validate_test_freeze(self.root, config, self.metas['dev_static'])

    def test_provisional_and_changed_parameters_rejected(self):
        config, _ = self.freeze()
        config['status'] = 'provisional'
        with self.assertRaisesRegex(ValueError, 'frozen'):
            validate_test_freeze(self.root, config, self.metas['test_static'])
        config['status'] = 'frozen'
        config['parameters']['history_weight'] = .9
        with self.assertRaisesRegex(ValueError, 'parameters or development evidence changed'):
            validate_test_freeze(self.root, config, self.metas['test_static'])

    def test_scene_leakage_and_duplicate_sources_rejected(self):
        self.rows[-1]['scene_group'] = 'dev_static'
        with self.assertRaisesRegex(ValueError, 'scene group cannot cross'):
            self.write_manifest()
            self.freeze()
        self.rows[-1]['scene_group'] = 'test_static'
        self.rows[-1]['notes'] = self.rows[0]['notes'].replace('cache/dev_static', 'cache/test_static')
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'Duplicate or changed source'):
            self.freeze()

    def test_missing_dev_and_unequal_model_inputs_rejected(self):
        removed = self.candidates['strong'].pop()
        with self.assertRaisesRegex(ValueError, 'all registered development clips'):
            self.freeze()
        self.candidates['strong'].append(removed)
        suite = self.root / removed
        meta = json.loads((suite / 'metadata.json').read_text())
        another = self.root / 'another_model/metadata.json'
        write_json(another, {'model': 'another'})
        meta['base_metadata_sha256'] = file_hash(another)
        write_json(suite / 'metadata.json', meta)
        self.receipt(suite)
        receipt = json.loads((suite / 'verification.json').read_text())
        receipt['input_metadata_sha256'].pop(f'inputs/{suite.name}/baseline/metadata.json')
        receipt['input_metadata_sha256']['another_model/metadata.json'] = file_hash(another)
        write_json(suite / 'verification.json', receipt)
        with self.assertRaisesRegex(ValueError, 'share model predictions'):
            self.freeze()

    def test_smoke_clip_cannot_be_used_as_development_evidence(self):
        path = self.root / 'cache/dev_static/metadata.json'
        meta = json.loads(path.read_text())
        meta['scope'] = 'pipeline_check_not_final_evaluation'
        write_json(path, meta)
        with self.assertRaisesRegex(ValueError, 'Only complete self-recorded'):
            self.freeze()

    def test_changed_result_review_and_manifest_rejected_after_freeze(self):
        config, _ = self.freeze()
        artifact = self.root / self.candidates['mild'][0] / 'E4/depths.npy'
        original = artifact.read_bytes()
        artifact.write_bytes(b'changed prediction')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            validate_test_freeze(self.root, config, self.metas['test_static'])
        artifact.write_bytes(original)
        review = json.loads(self.review.read_text())
        review['rationale'] = 'Changed decision after testing'
        write_json(self.review, review)
        with self.assertRaisesRegex(ValueError, 'evidence changed'):
            validate_test_freeze(self.root, config, self.metas['test_static'])
        self.rows[-1]['scene_group'] = 'another_test_group'
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'split changed'):
            validate_test_freeze(self.root, config, self.metas['test_static'])

    def test_unverified_or_incomplete_review_is_rejected(self):
        review = json.loads(self.review.read_text())
        review['candidate_observations']['mild']['dev_motion']['motion_response'] = ''
        write_json(self.review, review)
        with self.assertRaisesRegex(ValueError, 'Record stability'):
            self.freeze()
        review['candidate_observations']['mild']['dev_motion']['motion_response'] = 'Observed'
        write_json(self.review, review)
        (self.root / self.candidates['mild'][0] / 'verification.json').unlink()
        with self.assertRaises(FileNotFoundError):
            self.freeze()


if __name__ == '__main__':
    unittest.main()
