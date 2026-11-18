"""Scientific invariants for the controlled frozen-model refiner."""
from pathlib import Path
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from p3_depth.residual.model import Refiner, corrected_depth, loss_function
from p3_depth.residual.data import features_for, fit_scale
from p3_depth.evaluation.course import score_course_nyu


class ResidualStudyTests(unittest.TestCase):
    def test_zero_initialized_methods_preserve_metric_base(self):
        base = torch.linspace(.01, 12., 2*32*48).reshape(2, 1, 32, 48)
        for architecture in ('cnn', 'mlp', 'unet', 'legacy_unet'):
            model = Refiner(4, architecture)
            residual = model(torch.randn(2, 4, 16, 24))
            for kind in ('additive', 'log', 'bounded_log'):
                torch.testing.assert_close(corrected_depth(base, residual, kind), base, rtol=0, atol=0)

    def test_course_loss_matches_independent_numpy_metric(self):
        rng = np.random.default_rng(5)
        truth = rng.uniform(.1, 9.8, (480, 640)).astype('float32')
        pred = truth*rng.uniform(.5, 1.7, truth.shape).astype('float32')
        mask = np.zeros(truth.shape, bool)
        mask[45:471, 41:601] = True
        actual = loss_function(torch.from_numpy(pred)[None,None], torch.from_numpy(truth)[None,None], torch.from_numpy(mask)[None,None])
        expected = score_course_nyu(pred, truth)['abs_rel']
        self.assertAlmostEqual(float(actual), expected, places=6)

    def test_gt_outside_mask_cannot_affect_loss_or_gradient(self):
        pred = torch.full((1, 1, 8, 9), 3., requires_grad=True)
        truth = torch.full_like(pred, 2.)
        mask = torch.zeros_like(pred, dtype=torch.bool)
        mask[:,:,2:6,2:7] = True
        altered = torch.where(mask, truth, torch.full_like(truth, 10000.))
        for kind in ('abs_rel', 'gradient', 'boundary'):
            a = loss_function(pred, truth, mask, kind, mask)
            b = loss_function(pred, altered, mask, kind, mask)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        loss_function(pred, truth, mask, 'gradient').backward()
        self.assertTrue(bool(torch.all(pred.grad[~mask] == 0)))

    def test_scale_fitted_on_fit_only_and_correct_units(self):
        base = np.array([np.full((8, 9), 4.), np.full((8, 9), 4.)], dtype=np.float32)
        gt = np.array([np.full((8, 9), 2.), np.full((8, 9), 9.)], dtype=np.float32)
        mask = np.ones_like(base, bool)
        a = fit_scale(base, gt, mask, [0])['scale']
        gt[1] = .1
        b = fit_scale(base, gt, mask, [0])['scale']
        self.assertAlmostEqual(a, .5, places=4)
        self.assertEqual(a, b)

    def test_inference_inputs_do_not_read_gt(self):
        pack = {'original': np.ones((2, 16, 24), np.float32)*2, 'flip': np.ones((2,16,24), np.float32)*3,
                'rgb': np.zeros((2,3,8,12), np.float32), 'confidence': np.ones((2,8,12), np.float32)}
        # No truth field at all; every non-internal/non-edge feature must still work.
        for name in ('rgb_depth', 'depth_only', 'flip', 'confidence', 'coordinates', 'flip_disagreement', 'legacy'):
            value, _ = features_for(pack, {'inputs': name}, pack['original'], [8,12])
            self.assertTrue(np.isfinite(value).all())


if __name__ == '__main__':
    unittest.main()
