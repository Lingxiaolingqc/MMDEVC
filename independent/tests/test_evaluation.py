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


def test_final_evaluation_refuses_overwriting_a_previous_test_run(tmp_path):
    output=tmp_path/'test';output.mkdir();(output/'summary.json').write_text('{}')
    with pytest.raises(FileExistsError):
        evaluation.run_final_evaluation(tmp_path/'source',tmp_path/'baseline',tmp_path/'vda',tmp_path/'final.json',output)


def test_changed_fusion_flow_settings_do_not_reuse_evaluator_flow():
    from depth_project.records import Frame
    rgb=np.random.default_rng(3).integers(0,256,(48,64,3),np.uint8)
    frames=[Frame(0,0,rgb),Frame(1,0.04,rgb)]
    reference=np.ones((1,48,64,2),np.float32)
    forward,backward=evaluation.fusion_flows(frames,reference,reference,{'backend':'farneback','max_res':320})
    assert np.median(np.abs(forward))<0.01
    assert np.median(np.abs(backward))<0.01
