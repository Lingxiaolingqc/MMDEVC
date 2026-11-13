import numpy as np
import pytest
import torch
from filelock import FileLock, Timeout

from depth_project import metric_training as training
from depth_project.residual_refiner import make_features


def test_scaled_preparation_matches_original_feature_construction():
    rng = np.random.default_rng(7)
    rgb = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    original = rng.uniform(0.1, 5, (480, 640)).astype(np.float32)
    flipped = original * 1.1
    base = 0.75 * original + 0.25 * flipped
    confidence = np.ones_like(base)
    raw_features = make_features(rgb, original, flipped, confidence, base)
    actual = training.scaled_features(raw_features[None], base[None], 0.94)
    expected = make_features(rgb, original, flipped, confidence, base * 0.94)
    np.testing.assert_allclose(actual[0], expected, atol=2e-7)


def test_refiner_rgb_augmentation_preserves_geometry_channels_and_input():
    features = torch.rand(2, 9, 32, 40)
    original = features.clone()
    torch.manual_seed(42)
    output = training.augment_refiner_rgb(features)
    assert torch.equal(features, original)
    assert torch.equal(output[:, 3:], original[:, 3:])
    assert not torch.equal(output[:, :3], original[:, :3])
    assert output[:, :3].abs().max() <= 1


def test_atomic_json_retries_a_transient_windows_reader_lock(tmp_path, monkeypatch):
    from pathlib import Path
    import json

    original = Path.replace
    calls = []

    def temporarily_locked(path, target):
        calls.append(target)
        if len(calls) == 1:
            raise PermissionError("file is temporarily held by a reader")
        return original(path, target)

    monkeypatch.setattr(Path, "replace", temporarily_locked)
    monkeypatch.setattr(training.time, "sleep", lambda seconds: None)
    target = tmp_path / "summary.json"
    training.write_json(target, {"complete": True})
    assert json.loads(target.read_text()) == {"complete": True}
    assert len(calls) == 2


def test_training_entrypoint_records_subset_and_rejects_overwrite(tmp_path):
    torch.set_num_threads(4)
    base = np.full((3, 480, 640), 2, np.float32)
    truth = base.copy()
    truth[:, :, 320:] = 3
    pack = {
        "ids": [0, 1, 2], "scenes": {0: "a", 1: "b", 2: "c", 3: "test"},
        "base": base, "truth": truth,
        "features": np.zeros((3, 9, 32, 40), np.float32),
        "test_ids": [3], "identities": {"fixture": "synthetic"},
    }
    config = {"fit_ids": [0, 1], "validation_ids": [2], "seed": 7,
              "max_steps": 2, "max_epochs": 10, "evaluate_every": 1,
              "minimum_steps": 0, "patience": 2, "batch_size": 1}
    result = training.train_candidate(pack, config, tmp_path, device="cpu")
    assert result["completed_steps"] == 2
    assert result["fit_ids"] == [0, 1]
    assert result["validation_ids"] == [2]
    assert (tmp_path / "best.safetensors").exists()
    assert training.train_candidate(pack, config, tmp_path, device="cpu")["configuration_sha256"] == result["configuration_sha256"]
    with pytest.raises(ValueError, match="identity"):
        training.train_candidate(pack, {**config, "seed": 8}, tmp_path, device="cpu")
    (tmp_path / "summary.json").unlink()
    resumed = training.train_candidate(pack, config, tmp_path, device="cpu")
    assert resumed["checkpoint_sha256"] == result["checkpoint_sha256"]
    assert resumed["history"] == result["history"]
    (tmp_path / "summary.json").unlink()
    (tmp_path / "best.safetensors").write_bytes(b"interrupted external artifact")
    repaired = training.train_candidate(pack, config, tmp_path, device="cpu")
    assert repaired["checkpoint_sha256"] == result["checkpoint_sha256"]
    with FileLock(str(tmp_path / "run.lock"), timeout=0):
        with pytest.raises(Timeout):
            training.train_candidate(pack, config, tmp_path, device="cpu")
    (tmp_path / "best.safetensors").write_bytes(b"damaged checkpoint")
    with pytest.raises(ValueError, match="checkpoint"):
        training.train_candidate(pack, config, tmp_path, device="cpu")
    early_folder = tmp_path / "early"
    early_config = {**config, "max_steps": 4, "patience": 0}
    early = training.train_candidate(pack, early_config, early_folder, device="cpu")
    assert early["completed_steps"] == 1
    (early_folder / "summary.json").unlink()
    early_resumed = training.train_candidate(pack, early_config, early_folder, device="cpu")
    assert early_resumed["completed_steps"] == early["completed_steps"]
