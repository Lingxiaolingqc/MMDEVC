import json

import numpy as np
import pytest

from depth_project.metric_feature_cache import attach_depth_features


def test_prepared_features_keep_image_order_and_reject_changed_identities(tmp_path):
    source = tmp_path / "runs/metric-research/foundation-default"
    source.mkdir(parents=True)
    identities = {"ids": [2, 4], "dataset_sha256": "data", "split_sha256": "split", "checkpoint_sha256": "weights"}
    manifest = {"completed": 2, "identities": identities, "processed_hw": [490, 644], "feature_shape": [4, 2, 3]}
    (source / "manifest.json").write_text(json.dumps(manifest))
    for index in [2, 4]:
        np.savez(source / f"prediction_{index:04d}.npz", image_id=index, depth_features=np.full((4, 2, 3), index, np.float16))
    pack = {"ids": [2, 4], "identities": {key: identities[key] for key in ["dataset_sha256", "split_sha256", "checkpoint_sha256"]}}
    result = attach_depth_features(tmp_path, pack)
    np.testing.assert_array_equal(result["depth_features"][:, 0, 0, 0], [2, 4])
    assert "depth_features_sha256" in result["identities"]
    assert "depth_features_sha256" not in pack["identities"]
    assert attach_depth_features(tmp_path, pack)["identities"] == result["identities"]
    with pytest.raises(ValueError, match="identities"):
        attach_depth_features(tmp_path, {**pack, "identities": {**pack["identities"], "dataset_sha256": "changed"}})
