import numpy as np
import pytest
from depth_project import warp


def test_identity_warp():
    previous=np.array([[1,2,3],[4,5,6]],np.float32)
    actual,valid=warp.warp_previous(previous,np.zeros((2,3,2),np.float32))
    np.testing.assert_array_equal(actual,previous)
    assert valid.all()


def test_one_pixel_translation_and_disoccluded_border():
    previous=np.array([[1,2,3],[4,5,6]],np.float32)
    backward=np.zeros((2,3,2),np.float32); backward[:,:,0]=-1
    actual,valid=warp.warp_previous(previous,backward)
    np.testing.assert_array_equal(actual[:,1:],[[1,2],[4,5]])
    assert not valid[:,0].any()
    assert valid[:,1:].all()


def test_fractional_bilinear_sample():
    previous=np.array([[0,10],[20,30]],np.float32)
    backward=np.zeros((2,2,2),np.float32); backward[0,0]=[0.5,0.5]
    actual,valid=warp.warp_previous(previous,backward)
    assert valid[0,0]
    assert actual[0,0]==15


def test_nan_flow_and_nan_source_are_invalid():
    previous=np.array([[1,np.nan],[3,4]],np.float32)
    backward=np.zeros((2,2,2),np.float32); backward[1,0]=[np.nan,0]
    actual,valid=warp.warp_previous(previous,backward)
    assert not valid[0,1] and not valid[1,0]
    assert valid[0,0] and valid[1,1]
    assert np.isfinite(actual).all()


def test_forward_backward_consistency():
    forward=np.zeros((2,4,2),np.float32); forward[:,:,0]=1
    backward=-forward
    valid=warp.flow_consistency(forward,backward,{'alpha':0.01,'beta':0.5})
    assert not valid[:,0].any()
    assert valid[:,1:].all()
    backward[:,:,0]=-3
    assert not warp.flow_consistency(forward,backward,{'alpha':0.01,'beta':0.5}).any()
