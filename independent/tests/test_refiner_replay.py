import importlib.util
from pathlib import Path

import numpy as np
from safetensors.torch import save_file

from depth_project.records import sha256_file
from depth_project.residual_refiner import ResidualRefiner


def test_saved_refiner_predictions_preserve_ids_features_and_metric_scale(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/analyze_metric_learning_curves.py"
    spec = importlib.util.spec_from_file_location("refiner_analysis", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    settings = {"widths": [4, 8, 12, 16], "feature_channels": 4}
    model = ResidualRefiner(**settings)
    checkpoint = tmp_path / "model.safetensors"
    save_file(model.state_dict(), str(checkpoint))
    pack = {"ids": [0, 1], "features": np.zeros((2, 9, 16, 16), np.float16),
            "base": np.full((2, 480, 640), 2, np.float32), "truth": np.full((2, 480, 640), 2, np.float32),
            "scenes": {0: "fit", 1: "validation"}, "depth_features": np.ones((2, 4, 2, 2), np.float16)}
    summary = {"config": {"validation_ids": [1], "model": settings}, "scale": 1,
               "checkpoint_sha256": sha256_file(checkpoint)}
    output = tmp_path / "validation.npz"
    rows = module.evaluate_refiner(pack, summary, checkpoint, device="cpu", prediction_output=output)
    assert rows[0]["image_id"] == 1
    assert rows[0]["abs_rel"] == 0
    with np.load(output) as saved:
        np.testing.assert_array_equal(saved["image_ids"], [1])
        np.testing.assert_array_equal(saved["depth_m"], np.full((1, 480, 640), 2, np.float32))
