"""Protect depth representation, GT independence, and MATLAB axis/index contracts."""

from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np
from scipy.io import savemat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from p3_depth.data.nyuv2 import NYUv2Labeled, evaluation_mask, read_standard_split
from p3_depth.evaluation.depth import evaluate_depth, prepare_prediction


class DepthTests(unittest.TestCase):
    def setUp(self):
        self.gt = np.array([[1, 2, 4], [3, 5, 8]], dtype=np.float32)
        self.region = np.ones(self.gt.shape, dtype=bool)

    def prepare(self, raw, kind="metric_depth", unit="m", alignment="none"):
        return prepare_prediction(raw, self.gt, self.region, kind, unit, alignment)

    def test_direct_metric_units_and_no_gt_rescaling(self):
        depth, mask, metadata = self.prepare(self.gt * 2000, unit="mm")
        np.testing.assert_array_equal(depth, self.gt * 2)
        self.assertFalse(metadata["gt_used_for_alignment"])
        scores = evaluate_depth(depth, self.gt, self.region, mask)
        self.assertEqual(scores["depth_on_valid_predictions"]["abs_rel"], 1.)
        with self.assertRaises(ValueError):
            self.prepare(self.gt, alignment="median")

    def test_relative_depth_median_alignment(self):
        raw = self.gt * 4
        saved = raw.copy()
        depth, valid, metadata = self.prepare(raw, "relative_depth", "arbitrary", "median")
        np.testing.assert_array_equal(raw, saved)
        np.testing.assert_array_equal(depth, self.gt)
        self.assertEqual(metadata["scale"], .25)
        self.assertEqual(metadata["result_label"], "relative_aligned")
        self.assertEqual(evaluate_depth(depth, self.gt, self.region, valid)["depth_on_valid_predictions"]["delta1"], 1.)

    def test_inverse_scale_shift_is_fitted_in_inverse_domain(self):
        raw = (1/self.gt - .1)/2
        depth, valid, metadata = self.prepare(raw, "inverse_depth_like", "arbitrary", "inverse_scale_shift")
        np.testing.assert_allclose(depth, self.gt, rtol=1e-6)
        self.assertAlmostEqual(metadata["scale"], 2., places=6)
        self.assertAlmostEqual(metadata["shift"], .1, places=6)
        self.assertTrue(valid.all())

    def test_invalid_predictions_preserve_fixed_denominator(self):
        raw = self.gt.copy()
        raw[0] = [np.nan, 0., -1.]
        depth, valid, _ = self.prepare(raw)
        scores = evaluate_depth(depth, self.gt, self.region, valid)
        self.assertEqual(scores["gt_pixels"], 6)
        self.assertEqual(scores["coverage"], .5)
        self.assertEqual(scores["depth_on_valid_predictions"]["delta1"], 1.)
        self.assertEqual(scores["delta_full_gt_invalid_as_failure"]["delta1"], .5)
        empty, mask, _ = self.prepare(np.full_like(self.gt, np.nan))
        result = evaluate_depth(empty, self.gt, self.region, mask)
        self.assertIsNone(result["depth_on_valid_predictions"]["rmse_m"])
        self.assertEqual(result["delta_full_gt_invalid_as_failure"]["delta1"], 0.)

    def test_reject_wrong_representation_shape_and_degenerate_alignment(self):
        for raw, kind, unit, alignment in (
            (self.gt[:, :1], "metric_depth", "m", "none"),
            (self.gt.astype(np.uint8), "metric_depth", "m", "none"),
            (self.gt, "relative_depth", "m", "median"),
            (np.ones_like(self.gt), "inverse_depth_like", "arbitrary", "inverse_scale_shift"),
            (-1/self.gt, "inverse_depth_like", "arbitrary", "inverse_scale_shift"),
        ):
            with self.subTest(kind=kind, alignment=alignment), self.assertRaises(ValueError):
                self.prepare(raw, kind, unit, alignment)

    def test_alignment_does_not_fit_excluded_gt(self):
        region = self.region.copy()
        region[1] = False
        raw = self.gt * 3
        raw[1] *= 100
        depth, valid, metadata = prepare_prediction(raw, self.gt, region, "relative_depth", "arbitrary", "median")
        self.assertEqual(metadata["fit_pixels"], 3)
        self.assertAlmostEqual(metadata["scale"], 1/3)
        self.assertEqual(evaluate_depth(depth, self.gt, region, valid)["depth_on_valid_predictions"]["rmse_m"], 0.)


class NYUTests(unittest.TestCase):
    def setUp(self):
        base = (ROOT / "outputs/test-temp").resolve()
        base.mkdir(exist_ok=True, parents=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.path = Path(self.temp.name).resolve()
        assert self.path.is_relative_to(base)
        self.addCleanup(self.temp.cleanup)

    def test_matlab_axis_order_rgb_channels_and_one_based_index(self):
        path = self.path / "sample.mat"
        with h5py.File(path, "w") as f:
            rgb = f.create_dataset("images", shape=(1449, 3, 640, 480), dtype="u1", chunks=(1, 3, 64, 48))
            depth = f.create_dataset("depths", shape=(1449, 640, 480), dtype="f4", chunks=(1, 64, 48), fillvalue=2.)
            rgb[2, 0, 110, 70] = 255
            rgb[2, 2, 510, 370] = 128
            depth[2, 110, 70] = 3.5
        with NYUv2Labeled(path) as data:
            sample = data.load(3)
            self.assertEqual(sample.sample_id, "nyuv2_0003")
            self.assertEqual(sample.rgb.shape, (480, 640, 3))
            np.testing.assert_array_equal(sample.rgb[70, 110], [255, 0, 0])
            np.testing.assert_array_equal(sample.rgb[370, 510], [0, 0, 128])
            self.assertEqual(sample.depth_m[70, 110], 3.5)
            self.assertEqual(sample.depth_m[100, 100], 2.)
            with self.assertRaises(ValueError):
                data.load(0)

    def test_crop_bounds_and_inclusive_depth_interval(self):
        depth = np.ones((480, 640), dtype=np.float32)
        mask = evaluation_mask(depth)
        self.assertEqual(mask.sum(), (471-45)*(601-41))
        self.assertTrue(mask[45, 41] and mask[470, 600])
        self.assertFalse(mask[44, 41] or mask[471, 600] or mask[45, 601])
        depth[100, 100:105] = [0., .001, 10., 10.1, np.nan]
        np.testing.assert_array_equal(evaluation_mask(depth)[100, 100:105], [False, True, True, False, False])

    def test_standard_split_partition_and_duplicate_rejection(self):
        path = self.path / "splits.mat"
        values = {"trainNdxs": np.arange(1, 796)[:, None], "testNdxs": np.arange(796, 1450)[:, None]}
        savemat(path, values)
        splits = read_standard_split(path)
        self.assertEqual((len(splits["train"]), len(splits["test"])), (795, 654))
        values["testNdxs"][0] = 1
        savemat(path, values)
        with self.assertRaises(ValueError):
            read_standard_split(path)


if __name__ == "__main__":
    unittest.main()
