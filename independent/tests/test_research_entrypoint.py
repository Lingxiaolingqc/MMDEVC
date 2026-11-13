import json

import h5py
import numpy as np
import pytest

from depth_project import research_evaluation
from depth_project.records import DepthPrediction, sha256_file


@pytest.fixture
def setup(tmp_path, monkeypatch):
    data_folder = tmp_path / "data/nyu"
    data_folder.mkdir(parents=True)
    with h5py.File(data_folder / "nyu_depth_v2_labeled.mat", "w") as data:
        data["images"] = np.stack([np.full((3, 640, 480), value, np.uint8) for value in [60, 90, 120, 150]])
    (data_folder / "splits.mat").write_bytes(b"split")
    (tmp_path / "configs").mkdir()
    previous = tmp_path / "configs/metric-final.json"
    previous.write_text("previous frozen recipe")
    config = {"format_version": 2, "selection_status": "frozen after development selection and full-development refit",
              "dataset_sha256": sha256_file(data_folder / "nyu_depth_v2_labeled.mat"),
              "split_sha256": sha256_file(data_folder / "splits.mat"), "test_ids": [2, 3],
              "selection_fit_ids": [0], "selection_validation_ids": [1],
              "refit_jobs": [{"fit_ids": [0, 1], "validation_ids": [], "max_epochs": 1}],
              "previous_config_sha256": sha256_file(previous), "foundation_checkpoint_sha256": "foundation",
              "selected": {"value": 1.8}, "pre_refit": {"value": 1.5}, "inference_source_sha256": {}}
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(config))
    metadata = {"kind": "metric_m", "model_id": "fake", "checkpoint_sha256": "weights",
                "preprocessing_id": "official", "precision": "float32"}
    selected_calls = []
    failure = {"enabled": False}

    class Predictor:
        device = "cpu"

        def __init__(self, root, specification, foundation):
            self.value = specification["value"]
            self.metadata = metadata

        def predict_rgb(self, rgb):
            if self.value == 1.8:
                selected_calls.append(int(rgb[0, 0, 0]))
                if failure["enabled"] and rgb[0, 0, 0] == 150:
                    failure["enabled"] = False
                    raise RuntimeError("interrupted")
            return DepthPrediction(np.full((480, 640), self.value, np.float32), **metadata)

    class Reference:
        def __init__(self, root):
            pass

        def predict_pair(self, rgb):
            return {"baseline_m": np.ones((480, 640), np.float32),
                    "previous_refiner_m": np.full((480, 640), 1.2, np.float32)}

    def evaluate(root, config_path, output):
        for index in [2, 3]:
            with np.load(output / f"prediction_{index:04d}.npz") as packet:
                assert int(packet["image_id"]) == index
                assert str(packet["configuration_sha256"]) == sha256_file(config_path)
                np.testing.assert_array_equal(packet["selected_m"], np.full((480, 640), 1.8, np.float32))
        return {"image_count": 2, "metrics": {}}

    monkeypatch.setattr(research_evaluation, "LoRADepthPredictor", Predictor)
    monkeypatch.setattr(research_evaluation, "MetricRefinerReference", Reference)
    monkeypatch.setattr(research_evaluation, "load_nyu_split", lambda path: {"development": np.array([0, 1]), "test": np.array([2, 3])})
    monkeypatch.setattr(research_evaluation, "run_provenance", lambda root: {})
    monkeypatch.setattr(research_evaluation, "evaluate_metric_packets", evaluate)
    return tmp_path, path, config, selected_calls, failure


def test_prediction_generation_uses_rgb_and_recovers_completed_packets(setup):
    root, path, config, calls, failure = setup
    output = root / "result"
    failure["enabled"] = True
    with pytest.raises(RuntimeError, match="interrupted"):
        research_evaluation.run_research_evaluation(root, path, output)
    first_hash = sha256_file(output / "prediction_0002.npz")
    result = research_evaluation.run_research_evaluation(root, path, output)
    assert result["image_count"] == 2
    assert calls.count(120) == 1
    assert sha256_file(output / "prediction_0002.npz") == first_hash
    assert (output / "summary.json").is_file()
    with pytest.raises(FileExistsError):
        research_evaluation.run_research_evaluation(root, path, output)


def test_unfrozen_or_incomplete_refit_recipe_is_rejected(setup):
    root, path, config, calls, failure = setup
    path.write_text(json.dumps({**config, "selection_status": "searching"}))
    with pytest.raises(ValueError, match="frozen"):
        research_evaluation.run_research_evaluation(root, path, root / "unfrozen")
    config["refit_jobs"][0]["fit_ids"] = [0]
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        research_evaluation.run_research_evaluation(root, path, root / "incomplete")


def test_resumed_evaluation_rejects_changed_packet_content(setup):
    root, path, config, calls, failure = setup
    output = root / "result"
    failure["enabled"] = True
    with pytest.raises(RuntimeError, match="interrupted"):
        research_evaluation.run_research_evaluation(root, path, output)
    packet_path = output / "prediction_0002.npz"
    with np.load(packet_path) as packet:
        changed = dict(packet)
    changed["baseline_m"] = changed["baseline_m"] + 0.1
    np.savez_compressed(packet_path, **changed)
    with pytest.raises(ValueError, match="content hash"):
        research_evaluation.run_research_evaluation(root, path, output)
