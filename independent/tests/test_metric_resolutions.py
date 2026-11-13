import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def test_resolution_comparison_rejects_a_mislabeled_source(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/analyze_metric_resolutions.py"
    spec = importlib.util.spec_from_file_location("resolution_analysis", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "__file__", str(tmp_path / "scripts/analyze_metric_resolutions.py"))
    identity = {"dataset_sha256": "dataset", "split_sha256": "split", "checkpoint_sha256": "foundation"}
    pack = {"ids": [0, 1], "fit_ids": [0], "validation_ids": [1], "identities": identity,
            "scenes": {0: "fit", 1: "validation"},
            "truth": np.stack([np.full((480, 640), value, np.float32) for value in [4, 6]])}
    monkeypatch.setattr(module, "load_development_pack", lambda root: pack)
    for label in ["default", "4", "9"]:
        folder = tmp_path / "runs/metric-research" / f"foundation-{label}"
        folder.mkdir(parents=True)
        manifest = {"completed": 2, "processed_hw": [490, 644],
                    "identities": {**identity, "ids": [0, 1], "resolution": label}}
        (folder / "manifest.json").write_text(json.dumps(manifest))
        for index in [0, 1]:
            np.savez(folder / f"prediction_{index:04d}.npz", image_id=index,
                     depth_m=np.full((480, 640), 2, np.float32))
    output = tmp_path / "results/metric-research"
    output.mkdir(parents=True)
    module.main()
    records = json.loads((output / "resolutions.json").read_text())["records"]
    assert records[1]["scale"] == pytest.approx(2.0)
    assert records[1]["abs_rel"] == pytest.approx(1 / 3)
    path = tmp_path / "runs/metric-research/foundation-9/manifest.json"
    changed = json.loads(path.read_text())
    changed["identities"]["resolution"] = "4"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="resolution"):
        module.main()
