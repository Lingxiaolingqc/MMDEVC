from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.torch import save_file
import torch

from depth_project import research_inference
from depth_project.records import sha256_file
from depth_project.residual_refiner import ResidualRefiner


def test_feature_inference_reproduces_identity_with_scaled_half_precision_features(tmp_path, monkeypatch):
    class Foundation:
        def infer(self, tensor):
            return {"depth": (tensor[:1].float() / 255 + 1)[None],
                    "confidence": torch.ones(1, 1, *tensor.shape[-2:]),
                    "depth_features": torch.ones(1, 2, 2, 3)}
    adapter = SimpleNamespace(model=Foundation(), device="cpu", metadata={"checkpoint_sha256": "base",
                               "preprocessing_id": "official", "precision": "float32"})
    monkeypatch.setattr(research_inference, "load_image_adapter", lambda *args: adapter)
    options = {"widths": [4, 4, 4, 4], "feature_channels": 2, "gated": True}
    path = tmp_path / "weights.safetensors"
    save_file(ResidualRefiner(**options).state_dict(), str(path))
    specification = {"model": options, "feature_hw": [16, 16], "flip_weight": 0.25,
                     "members": [{"seed": 1, "checkpoint": path.name, "sha256": sha256_file(path), "scale": 1.2}]}
    predictor = research_inference.FeatureDepthPredictor(tmp_path, specification, "base")
    rgb = np.arange(16 * 24 * 3, dtype=np.uint8).reshape(16, 24, 3)
    expected = (rgb[:, :, 0].astype(np.float32) / 255 + 1) * 1.2
    np.testing.assert_array_equal(predictor.predict_members(rgb)[0], expected)
    specification["members"][0]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="identity"):
        research_inference.FeatureDepthPredictor(tmp_path, specification, "base")


def test_composite_averages_each_seed_pair_before_the_seed_mean(monkeypatch):
    left = [np.full((2, 3), value, np.float32) for value in [1.111111, 2.222222, 4.444444]]
    right = [np.full((2, 3), value, np.float32) for value in [3.123456, 5.654321, 7.876543]]
    metadata = {"kind": "metric_m", "model_id": "mock", "checkpoint_sha256": "mock",
                "preprocessing_id": "official", "precision": "float32"}
    def child(values):
        return SimpleNamespace(predict_members=lambda rgb: values, metadata=metadata, model=None, device="cpu")
    monkeypatch.setattr(research_inference, "LoRADepthPredictor", lambda *args: child(left))
    monkeypatch.setattr(research_inference, "FeatureDepthPredictor", lambda *args: child(right))
    members = [{"seed": value} for value in [1, 2, 3]]
    spec = {"type": "lora-feature-mean", "lora": {"members": members}, "refiner": {"members": members}}
    predictor = research_inference.load_research_predictor(None, spec, "base")
    pairs = [((a.astype(np.float64) + b) / 2).astype(np.float32) for a, b in zip(left, right)]
    expected = np.mean(np.stack(pairs), axis=0, dtype=np.float64).astype(np.float32)
    np.testing.assert_array_equal(predictor.predict_rgb(np.zeros((2, 3, 3), np.uint8)).values, expected)
    spec["refiner"] = {"members": [{"seed": 4}]}
    with pytest.raises(ValueError, match="seed"):
        research_inference.load_research_predictor(None, spec, "base")
