import numpy as np
import torch

from depth_project.residual_refiner import ResidualRefiner, make_features, apply_residual


def test_initial_refiner_preserves_metric_prediction_exactly():
    model = ResidualRefiner().eval()
    features = torch.randn(2, 9, 48, 64)
    with torch.inference_mode():
        residual = model(features)
    assert residual.shape == (2, 1, 48, 64)
    assert torch.count_nonzero(residual) == 0
    depth = np.full((96, 128), 2., np.float32)
    np.testing.assert_array_equal(apply_residual(depth, residual[0, 0].numpy()), depth)


def test_features_exclude_ground_truth_and_remain_finite_for_zero_depth():
    rgb = np.zeros((96, 128, 3), np.uint8)
    depth = np.zeros((96, 128), np.float32)
    result = make_features(rgb, depth, depth, np.ones_like(depth), np.ones_like(depth), (48, 64))
    assert result.shape == (9, 48, 64)
    assert np.isfinite(result).all()


def test_residual_is_multiplicative_and_uses_full_resolution_base():
    depth = np.array([[1., 2.], [3., 4.]], np.float32)
    output = apply_residual(depth, np.full((1, 1), np.log(1.2), np.float32))
    np.testing.assert_allclose(output, depth * 1.2, rtol=1e-6)
