"""Meaningful invariants for selective relaxation, rejection, and causal history."""
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from p3_depth.temporal.refinement import refinement_step, refinement_sequence, fixed_regions
from p3_depth.temporal.fusion import fuse_step, fuse_sequence


class RefinementTests(unittest.TestCase):
    def setUp(self):
        self.p = dict(history_weight=.7, fb_relative=.01, fb_absolute=.5,
                      photo_max=.2, photo_sigma=.1, flow_sigma_px=1.,
                      depth_span_sigma=.1, edge_span_sigma=.1)
        self.s = dict(depth_sigma_factor=2., reliable_photo_max=.05,
                      reliable_fb_max_px=.5, edge_guard_normalized=.02, edge_guard_radius_px=2)
        self.current = np.ones((16, 20), np.float32)
        self.current[:, 10:] = 3
        self.history = self.current + .1
        self.f = dict(backward=np.zeros((16, 20, 2), np.float32),
                      forward=np.zeros((16, 20, 2), np.float32),
                      photo_residual=np.zeros((16, 20), np.float32),
                      fb_residual=np.zeros((16, 20), np.float32))

    def test_protected_step_matches_original_and_interior_matches_global(self):
        old, ow = refinement_step(self.current, self.history, self.p, self.f, self.s, "original")
        broad, bw = refinement_step(self.current, self.history, self.p, self.f, self.s, "global")
        selected, sw = refinement_step(self.current, self.history, self.p, self.f, self.s, "selective")
        mask = fixed_regions(self.current, self.f, self.p, self.s)["reliable"]
        self.assertTrue(mask[:, :5].all())
        self.assertFalse(mask[:, 8:12].any())
        np.testing.assert_array_equal(selected[mask], broad[mask])
        np.testing.assert_array_equal(selected[~mask], old[~mask])
        np.testing.assert_array_equal(sw[~mask], ow[~mask])
        self.assertTrue(np.all(sw[mask] > ow[mask]))

    def test_factor_one_exactly_recovers_original_recursive_sequence(self):
        base = np.stack([self.current, self.history, self.current + .2])
        f = {k: np.stack([v, v]) for k, v in self.f.items()}
        expected = fuse_sequence(base, np.isfinite(base), "E4", self.p, f, representation="inverse_depth_like")
        for mode in ("global", "selective"):
            actual = refinement_sequence(base, np.isfinite(base), self.p, f,
                                         {**self.s, "depth_sigma_factor": 1.}, mode)
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a, b)

    def test_hard_rejection_and_missing_current_are_not_filled(self):
        self.f["photo_residual"][:, :5] = .9
        self.current[0, 0] = np.nan
        for mode in ("global", "selective"):
            result, weights = refinement_step(self.current, self.history, self.p, self.f, self.s, mode)
            self.assertTrue(np.isnan(result[0, 0]))
            np.testing.assert_array_equal(result[1:, :5], self.current[1:, :5])
            self.assertTrue((weights[:, :5] == 0).all())

    def test_prefix_independent_of_future_and_mask_independent_of_history(self):
        base = np.stack([self.current, self.history, self.current + .2])
        f = {k: np.stack([v, v]) for k, v in self.f.items()}
        full = refinement_sequence(base, np.isfinite(base), self.p, f, self.s, "selective")
        prefix = refinement_sequence(base[:2], np.isfinite(base[:2]), self.p,
                                     {k: v[:1] for k, v in f.items()}, self.s, "selective")
        for a, b in zip(full, prefix):
            np.testing.assert_array_equal(a[:2], b)


if __name__ == "__main__":
    unittest.main()

