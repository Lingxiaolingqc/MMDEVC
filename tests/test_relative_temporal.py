"""Analytical checks for arbitrary relative values, timestamps and missing outputs."""

from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from p3_depth.evaluation.temporal import evaluate_relative_video


class RelativeTemporalTests(unittest.TestCase):
    def evaluate(self, values, valid=None, static=False):
        valid = np.isfinite(values) if valid is None else valid
        return evaluate_relative_video(values, valid, np.zeros((*values[1:].shape, 2), np.float32),
                                       np.ones(values[1:].shape, bool), [0, .5], static_scene=static)

    def test_known_shift_retained_and_positive_affine_unit_changes(self):
        first = np.tile(np.arange(6, dtype=np.float32), (4, 1)) - 2
        values = np.stack([first, first+2])
        scores, rows = self.evaluate(values, static=True)
        self.assertEqual(scores['temporal_l1_relative_on_valid'], 2)
        self.assertEqual(scores['temporal_change_rate_relative_per_s_on_valid'], 4)
        self.assertEqual(scores['frame_median_relative'], [.5, 2.5])
        self.assertEqual(scores['median_absolute_change_relative'], 2)
        self.assertEqual(scores['static_pixel_std_relative'], 1)
        transformed, _ = self.evaluate(values*3+7)
        self.assertEqual(transformed['temporal_l1_relative_on_valid'], 6)
        self.assertAlmostEqual(transformed['capped_normalized_l1_full_reference'],
                               scores['capped_normalized_l1_full_reference'])
        self.assertEqual(rows[0]['pair_coverage'], 1)

    def test_invalid_half_pays_fixed_reference_penalty(self):
        values = np.stack([np.tile(np.arange(6, dtype=np.float32), (4, 1))]*2)
        valid = np.ones_like(values, dtype=bool)
        valid[1, :, :3] = False
        values[1, :, :3] = np.nan
        scores, _ = self.evaluate(values, valid)
        self.assertEqual(scores['reference_pixels'], 24)
        self.assertEqual(scores['pair_coverage'], .5)
        self.assertEqual(scores['temporal_l1_relative_on_valid'], 0)
        self.assertEqual(scores['capped_normalized_l1_full_reference'], .5)

    def test_constant_zero_is_valid_but_normalized_span_is_undefined(self):
        scores, _ = self.evaluate(np.zeros((2, 4, 6), np.float32), static=True)
        self.assertEqual(scores['pair_coverage'], 1)
        self.assertEqual(scores['temporal_l1_relative_on_valid'], 0)
        self.assertEqual(scores['static_pixel_std_relative'], 0)
        self.assertIsNone(scores['capped_normalized_l1_full_reference'])
        missing, _ = self.evaluate(np.full((2, 4, 6), np.nan, np.float32))
        self.assertEqual(missing['pair_coverage'], 0)
        self.assertIsNone(missing['temporal_l1_relative_on_valid'])
        self.assertIsNone(missing['median_absolute_change_relative'])


if __name__ == '__main__':
    unittest.main()
