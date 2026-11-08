import json

import numpy as np
import pytest

from depth_project import cache


@pytest.mark.parametrize("field", ["checkpoint_sha256", "input_size", "precision", "preprocessing", "frame_ids", "upstream_commit"])
def test_cache_key_covers_inference_changes(field):
    config = {"checkpoint_sha256": "a", "input_size": 518, "precision": "fp16", "preprocessing": "official", "frame_ids": [0,1], "upstream_commit": "abc"}
    old = cache.cache_key("source", config)
    config[field] = "different"
    assert cache.cache_key("source", config) != old
    assert cache.cache_key("different-source", {}) != cache.cache_key("source", {})


def test_cache_roundtrip_and_frame_identity(tmp_path):
    values = np.array([[[1,2]], [[3,4]]], np.float32)
    path = tmp_path / "depth.npz"
    cache.save_predictions(path, values, {"key":"abc", "frame_ids":[0,1]})
    actual, metadata = cache.load_predictions(path, expected_key="abc", expected_frame_ids=[0,1])
    np.testing.assert_array_equal(actual, values)
    assert metadata["key"] == "abc"
    with pytest.raises(ValueError, match="key"):
        cache.load_predictions(path, expected_key="wrong")
    with pytest.raises(ValueError, match="frame"):
        cache.load_predictions(path, expected_frame_ids=[1,0])


def test_rejects_truncated_and_nonfinite_cache(tmp_path):
    path = tmp_path / "depth.npz"
    path.write_bytes(b"broken")
    with pytest.raises(ValueError, match="cache"):
        cache.load_predictions(path)
    with pytest.raises(ValueError, match="finite"):
        cache.save_predictions(path, np.array([[[np.nan]]]), {"frame_ids":[0]})


def test_rejects_incomplete_frame_count(tmp_path):
    with pytest.raises(ValueError, match="frame"):
        cache.save_predictions(tmp_path / "depth.npz", np.ones((2,3,4)), {"frame_ids":[0]})
