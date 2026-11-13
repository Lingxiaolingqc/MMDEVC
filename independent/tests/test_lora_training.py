import json

import h5py
import numpy as np
import pytest
import torch
from torch import nn

from depth_project import lora_training


class TinyFoundation(nn.Module):
    def __init__(self):
        super().__init__()
        self.pixel_encoder = nn.Module()
        self.pixel_encoder.block = nn.Module()
        self.pixel_encoder.block.attn = nn.Module()
        self.pixel_encoder.block.attn.qkv = nn.Linear(3, 9)
        self.register_buffer("constant", torch.tensor(2.0))


def tiny_prediction(model, rgb):
    values = rgb.float().mean(dim=(1, 2))[None] / 255
    update = model.pixel_encoder.block.attn.qkv(values).mean() * 0.1
    return (model.constant + update).expand(1, 1, 480, 640)


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    torch.manual_seed(1)
    model = TinyFoundation().eval()
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    images = np.full((4, 3, 640, 480), 100, dtype=np.uint8)
    folder = tmp_path / "data/nyu"
    folder.mkdir(parents=True)
    with h5py.File(folder / "nyu_depth_v2_labeled.mat", "w") as source:
        source["images"] = images
    cache = tmp_path / "runs/nyu-development"
    cache.mkdir(parents=True)
    for index in range(4):
        prediction = tiny_prediction(model, torch.from_numpy(images[index].transpose(0, 2, 1))).detach().numpy()[0, 0]
        np.savez(cache / f"prediction_{index:04d}.npz", depth_m=prediction)

    def load_adapter(*args, **kwargs):
        foundation = TinyFoundation().eval()
        foundation.load_state_dict(initial)
        return type("Adapter", (), {"model": foundation, "metadata": {"checkpoint_sha256": "foundation"}})()

    monkeypatch.setattr(lora_training, "load_image_adapter", load_adapter)
    monkeypatch.setattr(lora_training, "infer_differentiable", tiny_prediction)
    monkeypatch.setattr(lora_training, "run_provenance", lambda root: {})
    pack = {"ids": [0, 1, 2, 3], "fit_ids": [0, 1], "validation_ids": [2, 3],
            "test_ids": [4], "scenes": {index: str(index) for index in range(5)},
            "truth": np.stack([np.full((480, 640), 2.3 + index * 0.1, np.float32) for index in range(4)]),
            "identities": {"checkpoint_sha256": "foundation"}}
    config = {"rank": 4, "seed": 7, "fit_ids": [0, 1], "validation_ids": [2, 3],
              "max_epochs": 2, "schedule_epochs": 80, "learning_rate": 1e-4, "accumulation": 2}
    return tmp_path, pack, config


def test_training_checks_frozen_weights_and_reuses_verified_completion(experiment):
    root, pack, config = experiment
    output = root / "run"
    result = lora_training.train_lora_candidate(root, pack, config, output, device="cpu")
    assert result["frozen_parameters_unchanged"]
    assert result["completed_epochs"] == 2
    assert result["initial_maximum_baseline_difference_m"] == 0
    assert result["hardware"]["device"] == "cpu"
    assert result["hardware"]["precision"] == "float32"
    assert len(result["rows"]) == 2
    assert result["rows"][0]["image_id"] == 2
    assert lora_training.train_lora_candidate(root, pack, config, output, device="cpu") == result
    with pytest.raises(ValueError, match="configuration"):
        lora_training.train_lora_candidate(root, pack, {**config, "seed": 8}, output, device="cpu")
    (output / "best.safetensors").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="checkpoint"):
        lora_training.train_lora_candidate(root, pack, config, output, device="cpu")


def test_epoch_resume_repairs_best_and_reproduces_training(experiment):
    root, pack, config = experiment
    output = root / "run"
    expected = lora_training.train_lora_candidate(root, pack, config, output, device="cpu")
    (output / "summary.json").unlink()
    (output / "best.safetensors").write_bytes(b"damaged")
    actual = lora_training.train_lora_candidate(root, pack, config, output, device="cpu")
    assert actual["checkpoint_sha256"] == expected["checkpoint_sha256"]
    assert actual["best_abs_rel"] == expected["best_abs_rel"]
    assert actual["completed_epochs"] == 2


def test_fixed_development_refit_saves_final_epoch(experiment):
    root, pack, config = experiment
    config = {**config, "fit_ids": pack["ids"], "validation_ids": [], "max_epochs": 1}
    result = lora_training.train_lora_candidate(root, pack, config, root / "refit", device="cpu")
    assert result["best_epoch"] == 1
    assert result["score_scope"] == "fitting"
    assert result["config"]["fit_ids"] == pack["ids"]


def test_adapter_reload_rejects_missing_or_extra_weights():
    model = TinyFoundation()
    lora_training.configure_lora(model, 4)
    state = lora_training.adapter_state(model)
    lora_training.load_adapter_state(model, state)
    with pytest.raises(ValueError, match="names"):
        lora_training.load_adapter_state(model, {**state, "constant": torch.tensor(2.0)})
    with pytest.raises(ValueError, match="names"):
        lora_training.load_adapter_state(model, {})
