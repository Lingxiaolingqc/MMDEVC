"""Benchmark scope and missing-prediction accounting, independent of model weights."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from p3_depth.evaluation.benchmark import select_nyu, summarize, score_raw
from p3_depth.evaluation.depth import evaluate_depth


class BenchmarkTests(unittest.TestCase):
    def test_complete_standard_test_excludes_development(self):
        _, test = select_nyu(ROOT, 'test')
        _, dev = select_nyu(ROOT, 'dev')
        self.assertEqual(len(test), 654)
        self.assertEqual(len(dev), 3)
        self.assertFalse({r['sample_id'] for r in test} & {r['sample_id'] for r in dev})

    def test_empty_prediction_is_not_averaged_as_perfect(self):
        gt = np.ones((2, 3), dtype=np.float32)
        region = np.ones(gt.shape, dtype=bool)
        good = evaluate_depth(gt * 2, gt, region)
        missing = evaluate_depth(np.full_like(gt, np.nan), gt, region)
        result = summarize([{'metrics': good, 'inference_seconds': 1.}, {'metrics': missing, 'inference_seconds': 3.}])
        self.assertEqual(result['image_macro_metrics_on_valid']['rmse_m'], 1.)
        self.assertEqual(result['contributing_images']['rmse_m'], 1)
        self.assertEqual(result['zero_coverage_images'], 1)
        self.assertEqual(result['pixel_pooled_coverage'], .5)
        self.assertEqual(result['throughput_fps_excluding_io'], .5)

    def test_invalid_limits_do_not_select_empty_or_oversized_test(self):
        for limit in (0, -1, 655):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                select_nyu(ROOT, 'test', limit)

    def test_wrong_direction_alignment_counts_as_failure_without_dropping_image(self):
        gt=np.tile(np.linspace(1,4,640,dtype=np.float32),(480,1))
        raw=2-1/gt
        asset={'prediction_type':'inverse_depth_like','unit':'arbitrary','alignment':'inverse_scale_shift'}
        scores,transform=score_raw(raw,gt,np.ones_like(gt,dtype=bool),asset)
        self.assertEqual(transform['alignment_status'],'failed')
        self.assertEqual(scores['coverage'],0)
        self.assertIsNone(scores['depth_on_valid_predictions']['rmse_m'])
        self.assertEqual(scores['delta_full_gt_invalid_as_failure']['delta1'],0)
        summary=summarize([{'metrics':scores,'transformation':transform,'inference_seconds':1}])
        self.assertEqual(summary['images'],1);self.assertEqual(summary['alignment_failures'],1)
        self.assertEqual(summary['gt_pixels'],426*560)


if __name__ == '__main__':
    unittest.main()
