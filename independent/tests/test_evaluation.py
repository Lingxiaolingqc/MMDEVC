import numpy as np
import pytest
from depth_project import evaluation
from depth_project.cache import save_predictions


def test_source_mismatch_rejected_before_any_replay(tmp_path):
    source=tmp_path/'source.mp4';source.write_bytes(b'not-the-cached-input')
    cached=tmp_path/'depth.npz'
    save_predictions(cached,np.ones((2,3,4)),{'frame_ids':[0,1],'source_sha256':'different'})
    with pytest.raises(ValueError,match='source'):
        evaluation.prepare_evaluation(source,cached,tmp_path/'evaluation')


def test_selecting_flicker_winner_does_not_reward_blurred_edges():
    rows=[{'name':'blur','temporal':0.1,'edge_retention':0.5,'synthetic_settling_frames':1,'config':{}}, {'name':'good','temporal':0.8,'edge_retention':0.95,'synthetic_settling_frames':1,'config':{'history_cap':0.5}}, {'name':'lag','temporal':0.2,'edge_retention':0.98,'synthetic_settling_frames':6,'config':{}}]
    assert evaluation.select_configuration(rows)['name']=='good'
