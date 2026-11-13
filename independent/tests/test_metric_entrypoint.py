def test_metric_command_uses_requested_configuration_and_output(monkeypatch, tmp_path):
    from depth_project.cli import main
    from depth_project import metric_experiments
    called = []
    monkeypatch.setattr(metric_experiments, 'run_metric_evaluation', lambda root, config, output: called.append((config, output)))
    config = tmp_path / 'frozen.json'
    output = tmp_path / 'evaluation'
    assert main(['metric-improvement', '--config', str(config), '--output', str(output)]) == 0
    assert called == [(config, output)]


def test_research_configuration_uses_the_frozen_research_pipeline(monkeypatch, tmp_path):
    import json
    from depth_project import metric_experiments, research_evaluation

    config = tmp_path / "frozen.json"
    config.write_text(json.dumps({"format_version": 2}))
    output = tmp_path / "evaluation"
    called = []
    monkeypatch.setattr(research_evaluation, "run_research_evaluation",
                        lambda root, path, destination: called.append((root, path, destination)))
    metric_experiments.run_metric_evaluation(tmp_path, config, output)
    assert called == [(tmp_path, config, output)]
