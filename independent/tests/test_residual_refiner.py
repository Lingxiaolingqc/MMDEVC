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


def test_optional_feature_refiner_and_gate_preserve_zero_update_identity():
    model = ResidualRefiner(feature_channels=12, gated=True, dropout=0.1).eval()
    with torch.inference_mode():
        residual = model(torch.randn(2, 9, 49, 65), torch.randn(2, 12, 7, 9))
    assert residual.shape == (2, 1, 49, 65)
    assert torch.count_nonzero(residual) == 0


def test_narrow_refiner_keeps_legacy_state_names_and_reduces_parameters():
    original = ResidualRefiner()
    narrow = ResidualRefiner(widths=(8, 16, 32, 48), dropout=0.1)
    assert set(original.state_dict()) == set(narrow.state_dict())
    assert sum(parameter.numel() for parameter in narrow.parameters()) < 120000
    assert sum(parameter.numel() for parameter in original.parameters()) == 388001


def test_optional_modules_receive_gradients_after_residual_head_initialization():
    model = ResidualRefiner(feature_channels=12, gated=True)
    torch.nn.init.normal_(model.output.weight, std=0.01)
    model(torch.randn(1, 9, 32, 40), torch.randn(1, 12, 4, 5)).square().mean().backward()
    assert model.feature_projection.weight.grad.abs().sum() > 0
    assert model.gate.weight.grad.abs().sum() > 0


def test_larger_residual_bound_is_explicit_and_keeps_identity_start():
    model = ResidualRefiner(residual_bound=1.0).eval()
    values = torch.zeros(1, 9, 32, 40)
    with torch.inference_mode():
        assert torch.count_nonzero(model(values)) == 0
        model.output.bias.fill_(100)
        torch.testing.assert_close(model(values), torch.ones(1, 1, 32, 40))
