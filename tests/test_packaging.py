"""Archive integrity checks with labelled byte fixtures, not fabricated model results."""

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from p3_depth.models.runtime import file_hash
from p3_depth.packaging import collect_exports, write_archive


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')


class PackagingTests(unittest.TestCase):
    def setUp(self):
        base = ROOT/'outputs/test-temp'
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        scope = 'pipeline_check_not_final_evaluation'
        cache, run, render, demo = [self.root/x for x in ('cache', 'run', 'render', 'demo')]
        for path in (cache, run, render, demo):
            path.mkdir()
        (cache/'frames.csv').write_text('timestamp_s\n0\n0.1\n')
        write_json(cache/'metadata.json', {'clip_id': 'fixture', 'scope': scope, 'split': 'smoke',
            'scene_type': 'fixture', 'frames': 2, 'frames_sha256': 'fixture-only',
            'index_sha256': file_hash(cache/'frames.csv')})
        write_json(run/'metadata.json', {'model': 'E4', 'test_fixture': True})
        for path in (render/'comparison.mp4', render/'preview.png', demo/'demo.mp4', demo/'preview.png', demo/'point_trace.csv'):
            path.write_bytes(b'archive-integrity fixture, not actual media')
        write_json(render/'metadata.json', {'status': 'complete', 'scope': scope, 'frames': 2,
            'cache_frames_sha256': 'fixture-only', 'cache_index_sha256': file_hash(cache/'frames.csv'),
            'raw_predictions_modified': False, 'inputs': [{'model': 'E4', 'run': 'run',
                'metadata_sha256': file_hash(run/'metadata.json'), 'unit': 'm', 'low': 0, 'high': 10}],
            'video_sha256': file_hash(render/'comparison.mp4'), 'preview_sha256': file_hash(render/'preview.png')})
        write_json(demo/'metadata.json', {'status': 'complete', 'scope': scope, 'clip_id': 'fixture',
            'cache_metadata_sha256': file_hash(cache/'metadata.json'),
            'verification': {'decoded_timestamps_match': True, 'trace_matches_raw_predictions': True},
            'methods': [{'name': 'E4', 'metadata_sha256': file_hash(run/'metadata.json'), 'provisional': True}],
            'files': {name: file_hash(demo/name) for name in ('demo.mp4', 'preview.png', 'point_trace.csv')}})
        self.plan = {'schema_version': 1, 'scope': scope, 'clips': [{'clip_id': 'fixture',
            'cache': 'cache', 'renders': {'comparison': 'render'}, 'demo': 'demo'}]}

    def test_archive_roundtrip_hashes_and_explicit_file_set(self):
        (self.root/'personal_acceptance.md').write_text('Private acceptance document')
        files, manifest = collect_exports(self.root, self.plan)
        path = self.root/'depth_results_smoke.zip'
        result = write_archive(files, manifest, path)
        self.assertTrue(result['all_payload_sha256_match'])
        self.assertTrue(result['crc_checked'])
        self.assertFalse(result['course_submission_performed'])
        with zipfile.ZipFile(path) as archive:
            self.assertNotIn('personal_acceptance.md', archive.namelist())
            self.assertIn(b'PIPELINE CHECK ONLY', archive.read('README.txt'))
            stored = json.loads(archive.read('manifest.json'))
            self.assertEqual(set(stored['files']), set(files))
            self.assertEqual(len(archive.namelist()), len(files)+2)

    def test_changed_export_and_cross_clip_inputs_are_rejected(self):
        video = self.root/'render/comparison.mp4'
        original = video.read_bytes()
        video.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'hash changed'):
            collect_exports(self.root, self.plan)
        video.write_bytes(original)
        path = self.root/'render/metadata.json'
        meta = json.loads(path.read_text())
        meta['cache_index_sha256'] = 'another-clip'
        write_json(path, meta)
        with self.assertRaisesRegex(ValueError, 'RGB/time cache'):
            collect_exports(self.root, self.plan)

    def test_mixed_metric_ranges_and_missing_demo_are_rejected(self):
        shutil.copytree(self.root/'render', self.root/'another_render')
        path = self.root/'another_render/metadata.json'
        meta = json.loads(path.read_text())
        meta['inputs'][0]['high'] = 20
        write_json(path, meta)
        self.plan['clips'][0]['renders']['another'] = 'another_render'
        with self.assertRaisesRegex(ValueError, 'fixed display range'):
            collect_exports(self.root, self.plan)
        self.plan['clips'][0]['renders'].pop('another')
        self.plan['clips'][0].pop('demo')
        with self.assertRaisesRegex(ValueError, 'demonstration'):
            collect_exports(self.root, self.plan)

    def test_empty_formal_plan_and_smoke_final_name_cannot_pass(self):
        with self.assertRaisesRegex(ValueError, 'nonempty clip IDs'):
            collect_exports(self.root, {'schema_version': 1, 'scope': 'self_recorded_project', 'clips': []})
        files, manifest = collect_exports(self.root, self.plan)
        with self.assertRaisesRegex(ValueError, 'smoke archive name'):
            write_archive(files, manifest, self.root/'depth_results.zip')
        path = self.root/'existing.zip'
        path.write_bytes(b'preserve this original')
        with self.assertRaises(FileExistsError):
            write_archive(files, manifest, path)
        self.assertEqual(path.read_bytes(), b'preserve this original')


if __name__ == '__main__':
    unittest.main()
