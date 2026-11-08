import numpy as np
import pytest
from depth_project import flow


def test_resize_scales_xy_independently():
    small=np.ones((2,3,2),np.float32)
    enlarged=flow.resize_flow(small,(6,6))
    np.testing.assert_allclose(enlarged[:,:,0],2)
    np.testing.assert_allclose(enlarged[:,:,1],3)


def test_static_rgb_flow_has_no_motion():
    image=np.random.default_rng(3).integers(0,256,(48,64,3),np.uint8)
    forward,backward=flow.estimate_pair(image,image,{'backend':'farneback','max_res':64})
    assert forward.shape==(48,64,2)
    assert backward.shape==(48,64,2)
    assert np.median(np.abs(forward))<0.01


def test_motion_direction_on_translated_texture():
    previous=np.random.default_rng(4).integers(0,256,(64,96,3),np.uint8)
    current=np.zeros_like(previous); current[:,2:]=previous[:,:-2]
    forward,backward=flow.estimate_pair(previous,current,{'backend':'farneback','max_res':96})
    assert np.median(forward[10:-10,10:-10,0])==pytest.approx(2,abs=0.2)
    assert np.median(backward[10:-10,10:-10,0])==pytest.approx(-2,abs=0.2)
