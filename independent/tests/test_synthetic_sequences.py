import numpy as np
from depth_project import fusion


def test_two_frame_average_reduces_independent_static_noise():
    rng=np.random.default_rng(42)
    current=10+rng.normal(size=(64,64))
    history=10+rng.normal(size=(64,64))
    actual=fusion.fuse_step(current,history,np.ones((64,64),bool),np.ones((64,64)),0.5)
    assert np.mean((actual-10)**2)<np.mean((current-10)**2)


def test_translating_edge_and_disocclusion_use_current_truth():
    # Backward alignment already places the old edge at its correct current pixel.
    truth=np.array([[2,2,10,10]],np.float32)
    aligned=np.array([[2,2,10,99]],np.float32)
    valid=np.array([[True,True,True,False]])
    actual=fusion.fuse_step(truth,aligned,valid,np.ones((1,4)),0.75)
    np.testing.assert_array_equal(actual,truth)


def test_true_depth_step_exposes_ema_lag():
    current=np.array([[10]],np.float32)
    history=np.array([[2]],np.float32)
    actual=fusion.fuse_step(current,history,np.ones((1,1),bool),np.ones((1,1)),0.75)
    assert actual[0,0]==4
    assert abs(actual[0,0]-10)==6
