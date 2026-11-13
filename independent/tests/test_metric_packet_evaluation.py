import json

import h5py
import numpy as np
import pytest

from depth_project import metric_packet_evaluation as evaluation
from depth_project.records import sha256_file


@pytest.fixture
def packets(tmp_path, monkeypatch):
    folder = tmp_path / "data/nyu"
    folder.mkdir(parents=True)
    with h5py.File(folder / "nyu_depth_v2_labeled.mat", "w") as source:
        source["depths"] = np.full((2, 640, 480), 2.0, np.float32)
        reference = source.create_dataset("scenes", (1, 2), dtype=h5py.ref_dtype)
        for index in range(2):
            scene = source.create_dataset(f"scene{index}", data=np.array([[ord(letter)] for letter in f"room{index}"]))
            reference[0, index] = scene.ref
    (folder / "splits.mat").write_bytes(b"official split")
    config = {"dataset_sha256": sha256_file(folder / "nyu_depth_v2_labeled.mat"),
              "split_sha256": sha256_file(folder / "splits.mat"), "test_ids": [0, 1]}
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(evaluation, "load_nyu_split", lambda path: {"test": np.array([0, 1])})
    output = tmp_path / "evaluation"
    output.mkdir()
    for index in range(2):
        np.savez(output / f"prediction_{index:04d}.npz", image_id=index,
                 configuration_sha256=sha256_file(path),
                 baseline_m=np.full((480, 640), 1.0, np.float32),
                 previous_refiner_m=np.full((480, 640), 1.5, np.float32),
                 pre_refit_m=np.full((480, 640), 1.75, np.float32),
                 selected_m=np.full((480, 640), 2.0, np.float32))
    return tmp_path, path, output


def test_saved_float_predictions_recompute_paired_accuracy(packets):
    root, config, output = packets
    result = evaluation.evaluate_metric_packets(root, config, output)
    assert result["metrics"]["baseline"]["abs_rel"] == pytest.approx(0.5)
    assert result["metrics"]["selected"]["abs_rel"] == 0
    assert result["comparisons"]["baseline_to_selected"]["relative_abs_rel_reduction_percent"] == 100
    assert result["comparisons"]["previous_refiner_to_selected"]["improved_images"] == 2
    assert (output / "selected-per-image.csv").is_file()


def test_saved_packet_configuration_and_image_identity_are_required(packets):
    root, config, output = packets
    path = output / "prediction_0001.npz"
    with np.load(path) as packet:
        values = dict(packet)
    np.savez(path, **{**values, "configuration_sha256": "other"})
    with pytest.raises(ValueError, match="configuration"):
        evaluation.evaluate_metric_packets(root, config, output)
    np.savez(path, **{**values, "image_id": 0})
    with pytest.raises(ValueError, match="image"):
        evaluation.evaluate_metric_packets(root, config, output)
