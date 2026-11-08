import numpy as np
import pytest
from depth_project import fusion
from depth_project.records import Frame,DepthPrediction


def prediction(value):
    return DepthPrediction(np.full((24,32),value,np.float32),'relative','fixture','abc','identity','fp32')


def test_fusion_hand_calculation_and_invalid_history():
    current=np.array([[10,10]],np.float32)
    previous=np.array([[12,np.nan]],np.float32)
    actual=fusion.fuse_step(current,previous,np.array([[True,False]]),np.ones((1,2)),0.25)
    np.testing.assert_array_equal(actual,[[10.5,10]])


def test_zero_cap_preserves_current_and_weights_are_bounded():
    current=np.array([[10,20]],np.float32)
    aligned=np.array([[12,30]],np.float32)
    np.testing.assert_array_equal(fusion.fuse_step(current,aligned,np.ones((1,2),bool),np.array([[5,np.nan]]),0),current)
    actual=fusion.fuse_step(current,aligned,np.ones((1,2),bool),np.array([[5,-1]]),0.5)
    np.testing.assert_array_equal(actual,[[11,20]])
    with pytest.raises(ValueError):
        fusion.fuse_step(current,aligned,np.ones((1,2),bool),np.ones((1,2)),1.5)


def test_scene_cut_resets_exactly_and_frame_order_is_checked():
    first=Frame(0,0,np.zeros((24,32,3),np.uint8))
    output,state,_=fusion.update(first,prediction(12),None,{'method':'adaptive','history_cap':0.75})
    second=Frame(1,1,np.full((24,32,3),255,np.uint8))
    output,state,diagnostics=fusion.update(second,prediction(10),state,{'method':'adaptive','history_cap':0.75})
    np.testing.assert_array_equal(output.values,10)
    assert diagnostics['reset']=='scene_cut'
    with pytest.raises(ValueError,match='consecutive'):
        fusion.update(first,prediction(12),state,{'method':'adaptive'})


def test_flat_depth_has_finite_confidence():
    rgb=np.zeros((24,32,3),np.uint8)
    _,state,_=fusion.update(Frame(0,0,rgb),prediction(10),None,{'method':'adaptive'})
    result,_,diagnostics=fusion.update(Frame(1,0.04,rgb),prediction(10),state,{'method':'adaptive'})
    assert np.isfinite(result.values).all()
    assert np.isfinite(diagnostics['history_weight']).all()
    np.testing.assert_array_equal(result.values,10)


def test_invalid_current_does_not_get_silently_dropped():
    with pytest.raises(ValueError,match='finite'):
        fusion.fuse_step(np.array([[np.nan]]),np.ones((1,1)),np.ones((1,1),bool),np.ones((1,1)),0.5)
