import pytest
from depth_project import experiments


def test_missing_stereo_pairs_cannot_produce_a_summary(tmp_path):
    with pytest.raises(ValueError,match='15'):
        experiments.run_stereo(tmp_path/'empty',tmp_path/'output')
    assert not (tmp_path/'output'/'summary.json').exists()
