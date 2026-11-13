import json

import numpy as np
import pytest
from safetensors.torch import save_file
import torch

from depth_project import metric_reference
from depth_project.records import sha256_file
from depth_project.residual_refiner import ResidualRefiner


def prepare_reference(root, monkeypatch):
    (root / "configs").mkdir()
    checkpoint = root / "reference.safetensors"
    save_file(ResidualRefiner().state_dict(), str(checkpoint))
    config = {"refiner_checkpoint": checkpoint.name, "refiner_sha256": sha256_file(checkpoint),
              "foundation_checkpoint_sha256": "foundation", "feature_hw": [16, 16],
              "simple_config": {"flip_weight": 0.25, "scale": 0.8, "radius": 0, "epsilon": 0.001}}
    (root / "configs/metric-final.json").write_text(json.dumps(config))

    class Foundation:
        calls = 0

        def infer(self, rgb):
            self.calls += 1
            depth = torch.full((1, 1, *rgb.shape[-2:]), 2.0 if self.calls % 2 else 3.0)
            return {"depth": depth, "confidence": torch.ones_like(depth)}

    metadata = {"kind": "metric_m", "model_id": "foundation", "checkpoint_sha256": "foundation",
                "preprocessing_id": "official", "precision": "float32"}
    adapter = type("Adapter", (), {"model": Foundation(), "device": "cpu", "metadata": metadata})()
    monkeypatch.setattr(metric_reference, "load_image_adapter", lambda *args: adapter)
    return config, adapter


def test_prior_method_reuses_its_original_blend_scale_and_resize(tmp_path, monkeypatch):
    config, adapter = prepare_reference(tmp_path, monkeypatch)
    predictor = metric_reference.MetricRefinerReference(tmp_path)
    rgb = np.full((8, 10, 3), 100, np.uint8)
    pair = predictor.predict_pair(rgb)
    np.testing.assert_array_equal(pair["baseline_m"], np.full((8, 10), 2.0, np.float32))
    np.testing.assert_allclose(pair["previous_refiner_m"], 1.8, rtol=1e-6)
    assert adapter.model.calls == 2
    assert predictor.predict_rgb(rgb).kind == "metric_m"


def test_reference_rejects_changed_original_weights(tmp_path, monkeypatch):
    config, adapter = prepare_reference(tmp_path, monkeypatch)
    (tmp_path / config["refiner_checkpoint"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint"):
        metric_reference.MetricRefinerReference(tmp_path)
