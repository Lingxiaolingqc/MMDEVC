"""Check signed correspondence, fixed-point units, actual matching and metric masks."""

from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p3_depth.data.middlebury import Calibration
from p3_depth.evaluation.stereo import evaluate_region
from p3_depth.models.stereo_sgbm import (
    StereoConfig, decode_disparity, left_right_consistency, predict_stereo,
)


def calibration(width=192, height=64, ndisp=16):
    cam0 = np.array([[100., 0., 50.], [0., 100., 30.], [0., 0., 1.]])
    cam1 = cam0.copy()
    cam1[0, 2] += 10
    return Calibration(cam0, cam1, 200., 10., width, height, ndisp)


class StereoTests(unittest.TestCase):
    def test_fixed_point_sentinel_and_valid_zero(self):
        raw = np.array([[-16, 0, 128, 129]], dtype=np.int16)
        np.testing.assert_allclose(decode_disparity(raw, 0), [[np.nan, 0, 8, 8.0625]], equal_nan=True)
        np.testing.assert_allclose(decode_disparity(np.array([[-256, -240, 0]], dtype=np.int16), -15),
                                   [[np.nan, -15, 0]], equal_nan=True)

    def test_lr_sign_bounds_and_missing_right_prediction(self):
        left = np.array([[2, 2, 2, 2, 2, np.nan]], dtype=np.float32)
        right = np.array([[-2, -3.5, np.nan, -2, -2, -2]], dtype=np.float32)
        valid, error = left_right_consistency(left, right, 1)
        np.testing.assert_array_equal(valid, [[False, False, True, False, False, False]])
        np.testing.assert_allclose(error, [[np.nan, np.nan, 0, 1.5, np.nan, np.nan]], equal_nan=True)

    def test_lr_nearest_sampling_and_inclusive_threshold(self):
        left = np.array([[np.nan, 0.5, 0.5, 3.]], dtype=np.float32)
        right = np.array([[-3., -1.5, -1.5, -1.5]], dtype=np.float32)
        valid, error = left_right_consistency(left, right, 1)
        np.testing.assert_array_equal(valid, [[False, True, True, True]])
        np.testing.assert_allclose(error[0, 1:], [1, 1, 0])

    def test_search_uses_conservative_calibration_bound(self):
        resolved = StereoConfig().resolved(calibration(ndisp=73))
        self.assertEqual(resolved["num_disparities"], 80)
        self.assertEqual(resolved["right_min_disparity"], -79)
        self.assertEqual((resolved["p1"], resolved["p2"]), (200, 800))
        for kwargs in ({"block_size": 4}, {"lr_max_diff_px": float("nan")}, {"uniqueness_ratio": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                StereoConfig(**kwargs)
        with self.assertRaises(ValueError):
            StereoConfig().resolved(calibration(width=16))

    def test_real_matcher_known_shift_and_calibrated_depth(self):
        rng = np.random.default_rng(13)
        texture = rng.integers(0, 256, (64, 192), dtype=np.uint8)
        left = np.repeat(texture[..., None], 3, axis=2)
        right = np.zeros_like(left)
        right[:, :-8] = left[:, 8:]
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
        pred = predict_stereo(left, right, calibration(), StereoConfig())
        region = np.zeros((64, 192), dtype=bool)
        region[8:-8, 32:-32] = True
        selected = region & pred.valid
        self.assertGreater(selected.sum() / region.sum(), .98)
        self.assertAlmostEqual(float(np.median(pred.disparity_px[selected])), 8, delta=.1)
        self.assertAlmostEqual(float(np.median(pred.depth_m[selected])), 20/18, delta=.01)
        self.assertEqual(pred.depth_m.dtype, np.float32)
        self.assertEqual(pred.valid.dtype, np.bool_)
        np.testing.assert_array_equal(np.isfinite(pred.depth_m), pred.valid)
        self.assertTrue(np.isnan(pred.disparity_px[~pred.valid]).all())

    def test_input_shape_and_dtype_rejected(self):
        image = np.zeros((64, 192, 3), dtype=np.uint8)
        for wrong in (image[:, :-1], image.astype(np.float32)):
            with self.assertRaises(ValueError):
                predict_stereo(image, wrong, calibration(), StereoConfig())


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.gt = np.array([[1., 2.], [4., 8.]], dtype=np.float32)
        self.disparity = np.full((2, 2), 10., dtype=np.float32)
        self.region = np.ones((2, 2), dtype=bool)

    def evaluate(self, depth=None, disparity=None, valid=None, region=None):
        return evaluate_region(self.gt if depth is None else depth,
                               self.disparity if disparity is None else disparity,
                               self.region if valid is None else valid, self.gt, self.disparity,
                               self.region if region is None else region)

    def test_perfect_prediction(self):
        result = self.evaluate()
        self.assertEqual(result["coverage"], 1)
        self.assertEqual(result["bad1_fraction_invalid_as_bad"], 0)
        for key, value in result["depth_on_valid_predictions"].items():
            self.assertEqual(value, 1 if key.startswith("delta") else 0)

    def test_scale_error_is_not_aligned_away(self):
        scores = self.evaluate(depth=self.gt * 2)["depth_on_valid_predictions"]
        self.assertEqual(scores["abs_rel"], 1)
        self.assertAlmostEqual(scores["rmse_m"], np.sqrt(85/4))
        self.assertAlmostEqual(scores["rmse_log"], np.log(2))
        self.assertEqual(scores["delta3"], 0)

    def test_invalid_coverage_and_strict_bad_threshold(self):
        valid = self.region.copy()
        valid[0, 0] = False
        disparity = np.array([[np.nan, 11], [12, 13]], dtype=np.float32)
        result = self.evaluate(disparity=disparity, valid=valid)
        self.assertEqual(result["gt_pixels"], 4)
        self.assertEqual(result["coverage"], .75)
        self.assertEqual(result["bad1_fraction_invalid_as_bad"], .75)
        self.assertEqual(result["bad2_fraction_invalid_as_bad"], .5)

    def test_zero_coverage_has_null_errors_and_full_bad_penalty(self):
        result = self.evaluate(valid=np.zeros_like(self.region))
        self.assertEqual(result["coverage"], 0)
        self.assertEqual(result["bad1_fraction_invalid_as_bad"], 1)
        self.assertTrue(all(value is None for value in result["depth_on_valid_predictions"].values()))

    def test_empty_region_shape_and_nonphysical_valid_prediction_fail(self):
        for kwargs in ({"region": np.zeros_like(self.region)}, {"depth": self.gt[:, :1]},
                       {"depth": np.zeros_like(self.gt)}, {"depth": np.full_like(self.gt, np.nan)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.evaluate(**kwargs)

    def test_region_does_not_include_excluded_gt_pixels(self):
        region = np.array([[True, False], [True, False]])
        depth = self.gt.copy()
        depth[:, 1] *= 100
        result = self.evaluate(depth=depth, region=region)
        self.assertEqual(result["gt_pixels"], 2)
        self.assertEqual(result["depth_on_valid_predictions"]["rmse_m"], 0)


if __name__ == "__main__":
    unittest.main()
