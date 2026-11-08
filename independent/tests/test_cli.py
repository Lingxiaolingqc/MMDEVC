import pytest
from depth_project.cli import main


def test_stereo_cli_uses_dataset_validation(tmp_path):
    with pytest.raises(ValueError,match='15'):
        main(['stereo','--data-root',str(tmp_path/'empty'),'--output',str(tmp_path/'output')])


def test_existing_benchmark_output_is_rejected_before_expensive_work(tmp_path):
    output=tmp_path/'runtime.json';output.write_text('preserve')
    with pytest.raises(FileExistsError):
        main(['benchmark','--input',str(tmp_path/'missing.mp4'),'--config',str(tmp_path/'missing.json'),'--output',str(output)])
    assert output.read_text()=='preserve'
