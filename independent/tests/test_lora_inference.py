import numpy as np
import pytest
from safetensors.torch import save_file
import torch
from torch import nn

from depth_project import lora_inference
from depth_project.lora_training import adapter_state, frozen_digest, load_adapter_state
from depth_project.metric_adaptation import configure_lora
from depth_project.records import sha256_file


class TinyFoundation(nn.Module):
    def __init__(self):
        super().__init__()
        self.pixel_encoder = nn.Module()
        self.pixel_encoder.block = nn.Module()
        self.pixel_encoder.block.attn = nn.Module()
        self.pixel_encoder.block.attn.qkv = nn.Linear(3, 9)

    def infer(self, rgb):
        values = rgb.float().mean(dim=(1, 2))[None] / 255
        depth = 2 + self.pixel_encoder.block.attn.qkv(values).mean() * 0.1
        return {"depth": depth.expand(1, 1, *rgb.shape[-2:])}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    def foundation():
        torch.manual_seed(1)
        return TinyFoundation().eval()

    def load(*args, **kwargs):
        return type("Adapter", (), {"model": foundation(), "device": "cpu",
                    "metadata": {"checkpoint_sha256": "foundation", "preprocessing_id": "official", "precision": "float32"}})()

    monkeypatch.setattr(lora_inference, "load_image_adapter", load)
    rgb = np.full((6, 8, 3), 100, np.uint8)
    manual = foundation()
    configure_lora(manual, 4)
    members, predictions = [], []
    for index, scale in enumerate([0.9, 1.1, 1.0]):
        state = adapter_state(manual)
        for key, values in state.items():
            values.fill_(0.01 * (index + 1))
        load_adapter_state(manual, state)
        with torch.no_grad():
            predictions.append((manual.infer(torch.from_numpy(rgb.transpose(2, 0, 1)))["depth"][0, 0] * scale).numpy())
        path = tmp_path / f"seed{index}.safetensors"
        save_file(state, str(path))
        members.append({"checkpoint": path.name, "sha256": sha256_file(path), "scale": scale})
    specification = {"rank": 4, "members": members}
    expected = np.mean(np.stack(predictions), axis=0, dtype=np.float64).astype(np.float32)
    return tmp_path, specification, rgb, expected


def test_rgb_only_ensemble_reloads_each_member_and_preserves_frozen_weights(setup):
    root, specification, rgb, expected = setup
    predictor = lora_inference.LoRADepthPredictor(root, specification, "foundation")
    frozen = frozen_digest(predictor.model)
    for _ in range(2):
        result = predictor.predict_rgb(rgb)
        np.testing.assert_array_equal(result.values, expected)
        assert result.kind == "metric_m"
        assert frozen_digest(predictor.model) == frozen


def test_inference_checks_foundation_and_adapter_hashes(setup):
    root, specification, rgb, expected = setup
    with pytest.raises(ValueError, match="foundation"):
        lora_inference.LoRADepthPredictor(root, specification, "wrong")
    specification["members"][0]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="checkpoint"):
        lora_inference.LoRADepthPredictor(root, specification, "foundation")
