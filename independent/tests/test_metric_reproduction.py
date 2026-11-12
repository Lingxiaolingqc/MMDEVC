import numpy as np
import pytest

from depth_project.metric_experiments import verify_frozen_dataset, historical_baseline_difference
from depth_project.records import sha256_file


@pytest.mark.parametrize('changed', ['dataset', 'split'])
def test_changed_frozen_input_is_rejected(tmp_path, changed):
    folder = tmp_path / 'data/nyu'
    folder.mkdir(parents=True)
    data = folder / 'nyu_depth_v2_labeled.mat'
    split = folder / 'splits.mat'
    data.write_bytes(b'dataset')
    split.write_bytes(b'split')
    config = {'dataset_sha256': sha256_file(data), 'split_sha256': sha256_file(split)}
    verify_frozen_dataset(tmp_path, config)
    (data if changed == 'dataset' else split).write_bytes(b'changed')
    with pytest.raises(ValueError, match='identity'):
        verify_frozen_dataset(tmp_path, config)


def test_historical_cache_is_optional_and_existing_comparison_is_measured(tmp_path):
    path = tmp_path / 'prediction.npz'
    prediction = np.ones((4, 5), np.float32)
    assert historical_baseline_difference(path, prediction) is None
    np.savez(path, depth_m=prediction + .2)
    assert historical_baseline_difference(path, prediction) == pytest.approx(.2)
