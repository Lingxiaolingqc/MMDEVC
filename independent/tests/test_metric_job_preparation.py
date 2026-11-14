import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def script(name, root):
    source = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    specification = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    module.__file__ = str(root / "scripts" / source.name)
    return module


def write(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_seed_jobs_share_weights_between_raw_and_flip_candidates(tmp_path):
    write(tmp_path, "results/metric-research/scene-fold-selection.json",
          {"advanced_candidates": ["lora-r8-high-flip", "lora-r8-high"]})
    write(tmp_path, "runs/nyu-development/manifest.json", {"fit_ids": [1, 2], "validation_ids": [3]})
    for name in ["control", "lora-r8-high"]:
        write(tmp_path, f"runs/metric-research/{name}-fold0/summary.json",
              {"config": {"fit_ids": [99], "validation_ids": [98], "seed": 320112,
                          "max_epochs": 80, "schedule_epochs": 80}})
    (tmp_path / ".cache/metric-research").mkdir(parents=True)
    script("prepare_metric_seeds", tmp_path).main()
    jobs = json.loads((tmp_path / ".cache/metric-research/seeds-lora-jobs.json").read_text())
    assert len(jobs) == 3
    assert {job["seed"] for job in jobs} == {320112, 320113, 320114}
    assert all(job["fit_ids"] == [1, 2] and job["validation_ids"] == [3] for job in jobs)
    assert all(job["schedule_epochs"] == 80 for job in jobs)


def test_refit_jobs_use_official_development_and_reject_mislabeled_seed(tmp_path, monkeypatch):
    source = {"config": {"seed": 320113, "fit_ids": [1, 2], "validation_ids": [3],
                          "max_epochs": 80, "schedule_epochs": 80}}
    selection = {"winner": "lora-r8-high-flip", "chosen_seeds": [320113],
                 "sources": {"selection-lora-r8-high-seed320113": source}}
    write(tmp_path, "results/metric-research/seed-selection.json", selection)
    write(tmp_path, "results/metric-research/scene-fold-selection.json",
          {"ranking": [{"candidate": "lora-r8-high", "fixed_refit_epochs": 5}]})
    write(tmp_path, "runs/nyu-development/manifest.json", {"fit_ids": [1, 2], "validation_ids": [3]})
    (tmp_path / ".cache/metric-research").mkdir(parents=True)
    module = script("prepare_metric_refits", tmp_path)
    monkeypatch.setattr(module, "load_nyu_split", lambda path: {"development": np.array([1, 2, 3]), "test": np.array([9])})
    module.main()
    jobs = json.loads((tmp_path / ".cache/metric-research/refit-lora-jobs.json").read_text())
    assert jobs[0]["fit_ids"] == [1, 2, 3]
    assert jobs[0]["validation_ids"] == []
    assert jobs[0]["max_epochs"] == 5
    assert jobs[0]["schedule_epochs"] == 80
    source["config"]["seed"] = 320112
    write(tmp_path, "results/metric-research/seed-selection.json", selection)
    (tmp_path / ".cache/metric-research/refit-lora-jobs.json").unlink()
    with pytest.raises(ValueError, match="seed"):
        module.main()
