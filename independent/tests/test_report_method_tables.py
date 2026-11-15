import importlib.util
import json
from pathlib import Path


def test_method_tables_expose_calibration_and_separate_seed_averaging(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = root / "scripts/write_metric_research_report_values.py"
    specification = importlib.util.spec_from_file_location("report_values", source)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    seeds = json.loads((root / "results/metric-research/seed-selection.json").read_text())
    search = json.loads((root / "results/metric-development-search.json").read_text())
    module.write_method_tables(tmp_path, seeds, search)
    methods = (tmp_path / "metric-research-method-table.tex").read_text()
    ablation = (tmp_path / "metric-research-component-table.tex").read_text()
    assert "LoRA + calibration" in methods
    assert "Per-seed pipeline AbsRel" in methods
    assert "0.08628" in methods and "0.08678" in methods and "0.08525" in methods
    assert "0.10843" in ablation and "0.08662" in ablation and "0.08548" in ablation
    assert "320113" in ablation
    assert "LR" not in methods and "Epochs" not in methods
    assert "19/40" not in methods
