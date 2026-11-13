import numpy as np
import pytest

from depth_project import adaptive_calibration as calibration


def test_ridge_recovers_image_specific_scale_from_fitting_examples():
    values = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.float64)
    scales = np.exp(values @ np.array([0.1, -0.1]) - 0.05)
    model = calibration.fit_ridge_scale(values, scales, alpha=1e-8)
    actual = calibration.predict_ridge_scale(model, values)
    np.testing.assert_allclose(actual, scales, atol=1e-8)
    assert model["fitting_count"] == 4


def test_ridge_normalization_uses_only_supplied_fitting_features():
    fitting = np.array([[0, 1], [1, 1], [2, 1]], dtype=np.float64)
    model = calibration.fit_ridge_scale(fitting, np.ones(3), alpha=1)
    np.testing.assert_array_equal(model["mean"], [1, 1])
    assert np.isfinite(calibration.predict_ridge_scale(model, [[1e8, 1]])).all()
    np.testing.assert_allclose(calibration.predict_ridge_scale(model, fitting), 1)


def test_global_features_are_fixed_length_and_finite():
    features = np.ones((8, 5, 7), np.float32)
    base = np.full((48, 64), 2, np.float32)
    result = calibration.global_features(features, base)
    assert result.shape == (15,)
    assert np.isfinite(result).all()
    with pytest.raises(ValueError):
        calibration.fit_ridge_scale(np.array([[float("nan")]]), [1], alpha=1)


def test_metric_global_adjustment_is_applied_to_predicted_scale():
    features = np.array([[0, 1], [1, 0], [2, 1]], np.float64)
    model = calibration.fit_ridge_scale(features, np.ones(3), alpha=1)
    model["metric_global_scale"] = 0.94
    np.testing.assert_allclose(calibration.predict_ridge_scale(model, features), 0.94)
