import unittest
import numpy as np

from p3_depth.evaluation.course import score_course_nyu, summarize_course


class CourseProtocolTests(unittest.TestCase):
    def test_crop_clip_and_perfect_prediction(self):
        gt = np.full((480, 640), 2., dtype=np.float32)
        pred = gt.copy()
        pred[0, 0] = np.nan  # Outside the fixed course crop.
        result = score_course_nyu(pred, gt)
        self.assertEqual(result['gt_pixels'], 426 * 560)
        self.assertEqual((result['abs_rel'], result['rmse_m'], result['delta1']), (0., 0., 1.))
        # Two valid GT pixels; others are excluded because GT is out of range.
        gt[:] = 0
        gt[45, 41], gt[470, 600] = 10., .001
        pred[45, 41], pred[470, 600] = 100., -5.
        result = score_course_nyu(pred, gt)
        self.assertEqual(result['gt_pixels'], 2)
        self.assertEqual((result['clipped_low_pixels'], result['clipped_high_pixels']), (1, 1))
        self.assertLess(result['rmse_m'], 1e-9)

    def test_nonfinite_prediction_cannot_reduce_denominator(self):
        gt = np.ones((480, 640), dtype=np.float32)
        pred = gt.copy()
        for bad in (np.nan, np.inf, -np.inf):
            pred[45, 41] = bad
            with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                score_course_nyu(pred, gt)

    def test_equal_image_weight_and_missing_image_rejected(self):
        rows = [dict(sample_id='a', gt_pixels=10, clipped_low_pixels=0, clipped_high_pixels=0,
                     abs_rel=0., rmse_m=0., delta1=1.),
                dict(sample_id='b', gt_pixels=1000, clipped_low_pixels=0, clipped_high_pixels=0,
                     abs_rel=1., rmse_m=2., delta1=0.)]
        result = summarize_course(rows, 2)
        self.assertEqual((result['abs_rel'], result['rmse_m'], result['delta1']), (.5, 1., .5))
        with self.assertRaises(ValueError):
            summarize_course(rows, 654)
        with self.assertRaises(ValueError):
            summarize_course([rows[0], rows[0]], 2)

    def test_delta_is_strict(self):
        gt = np.full((480, 640), 4., dtype=np.float32)
        pred = np.full_like(gt, 5.)
        result = score_course_nyu(pred, gt)
        self.assertEqual((result['abs_rel'], result['rmse_m'], result['delta1']), (.25, 1., 0.))


if __name__ == '__main__':
    unittest.main()
