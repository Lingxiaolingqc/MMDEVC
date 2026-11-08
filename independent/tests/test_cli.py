import pytest
from depth_project.cli import main


def test_stereo_cli_uses_dataset_validation(tmp_path):
    with pytest.raises(ValueError,match='15'):
        main(['stereo','--data-root',str(tmp_path/'empty'),'--output',str(tmp_path/'output')])
