import numpy as np
import pytest

from depth_project.refinement import fit_global_scale, guided_filter, blend_flip


def test_fit_scale_corrects_known_multiplicative_error():
    prediction = np.array([1., 2., 3., 4.])
    truth = prediction * 1.25
    assert fit_global_scale([prediction], [truth]) == pytest.approx(1.25)


def test_scale_fit_weights_images_equally():
    # First image has 100 pixels, second one pixel. Each image contributes equally.
    prediction = [np.ones(100), np.ones(1)]
    truth = [np.ones(100), np.full(1, .5)]
    assert fit_global_scale(prediction, truth) == pytest.approx(.5)


def test_scale_fit_rejects_invalid_support():
    with pytest.raises(ValueError):
        fit_global_scale([np.array([np.nan])], [np.array([1.])])
    with pytest.raises(ValueError):
        fit_global_scale([np.array([1.])], [np.array([0.])])


def test_guided_filter_preserves_constant_metric_depth():
    rng = np.random.default_rng(7)
    rgb = rng.integers(0, 256, (32, 40, 3), dtype=np.uint8)
    result = guided_filter(rgb, np.full((32, 40), 2.5, np.float32), 4, .01)
    np.testing.assert_allclose(result, 2.5, atol=1e-5)


def test_flip_blend_requires_already_unflipped_prediction():
    original = np.array([[1., 2., 3.]])
    restored = np.array([[1.2, 2.2, 3.2]])
    np.testing.assert_allclose(blend_flip(original, restored, .5), [[1.1, 2.1, 3.1]])
    np.testing.assert_array_equal(blend_flip(original, restored, 0), original)
    with pytest.raises(ValueError):
        blend_flip(original, restored, 2)
