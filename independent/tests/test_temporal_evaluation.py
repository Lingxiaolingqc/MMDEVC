import numpy as np
import pytest
from depth_project import metrics


def test_fixed_support_and_reference_scale():
    depths=np.array([[[1,2]],[[3,6]]],np.float32)
    backward=[np.zeros((1,2,2),np.float32)]
    masks=[np.array([[True,False]])]
    result=metrics.evaluate_video(depths,backward,masks,reference_scale=2,frame_ids=[0,1])
    assert result['aligned_mae_normalized']==1
    assert result['support_pixels']==1
    assert result['support_fraction']==0.5
    assert result['pair_count']==1
    # A candidate cannot change the evaluator's mask by having different values.
    another=metrics.evaluate_video(depths*2,backward,masks,reference_scale=2)
    assert another['support_pixels']==1
    assert another['aligned_mae_normalized']==2


def test_empty_support_is_unavailable_not_zero_error():
    result=metrics.evaluate_video(np.ones((2,2,2)),[np.zeros((2,2,2))],[np.zeros((2,2),bool)],1)
    assert result['aligned_mae_normalized'] is None
    assert result['support_pixels']==0


def test_frame_mismatch_and_nonfinite_prediction_are_errors():
    with pytest.raises(ValueError,match='frame|pair'):
        metrics.evaluate_video(np.ones((2,2,2)),[],[],1)
    with pytest.raises(ValueError,match='frame'):
        metrics.evaluate_video(np.ones((2,2,2)),[np.zeros((2,2,2))],[np.ones((2,2),bool)],1,frame_ids=[0,2])
    with pytest.raises(ValueError,match='finite'):
        metrics.evaluate_video(np.full((2,2,2),np.nan),[np.zeros((2,2,2))],[np.ones((2,2),bool)],1)


def test_frozen_depth_negative_control_can_score_zero_but_be_wrong():
    frozen=np.full((2,2,2),10,np.float32)
    result=metrics.evaluate_video(frozen,[np.zeros((2,2,2))],[np.ones((2,2),bool)],1)
    assert result['aligned_mae_normalized']==0
    truth=np.array([[[1,1],[1,1]],[[2,2],[2,2]]])
    assert np.mean((frozen-truth)**2)==72.5
