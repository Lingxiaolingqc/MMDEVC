import importlib.util
import json
from pathlib import Path


def checker_module():
    path = Path(__file__).resolve().parents[1] / "scripts/check_metric_research_packets.py"
    specification = importlib.util.spec_from_file_location("packet_checker", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_reproduction_check_keeps_published_evidence_and_honors_explicit_output(tmp_path):
    canonical = tmp_path / "results/independent-metric-check.json"
    canonical.parent.mkdir()
    canonical.write_text("published evidence", encoding="utf-8")
    run = tmp_path / "runs/reproduced"
    run.mkdir(parents=True)
    checker = checker_module()
    result = {"historical_reference_packets_checked": 0}
    destination = checker.save_check_result(run, result)
    assert destination == run / "independent-metric-check.json"
    assert canonical.read_text(encoding="utf-8") == "published evidence"
    assert json.loads(destination.read_text()) == result
    explicit = tmp_path / "results/fresh-check.json"
    assert checker.save_check_result(run, result, explicit) == explicit
    assert json.loads(explicit.read_text()) == result
