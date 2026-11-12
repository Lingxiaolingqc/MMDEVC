def test_metric_command_uses_requested_configuration_and_output(monkeypatch, tmp_path):
    from depth_project.cli import main
    from depth_project import metric_experiments
    called = []
    monkeypatch.setattr(metric_experiments, 'run_metric_evaluation', lambda root, config, output: called.append((config, output)))
    config = tmp_path / 'frozen.json'
    output = tmp_path / 'evaluation'
    assert main(['metric-improvement', '--config', str(config), '--output', str(output)]) == 0
    assert called == [(config, output)]
