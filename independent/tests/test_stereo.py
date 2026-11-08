import numpy as np
import pytest
from depth_project import stereo


def test_fixed_point_disparity_offset_and_meters():
    disp=stereo.fixed_point_to_pixels(np.array([160,-16],np.int16))
    np.testing.assert_array_equal(disp,[10,-1])
    depth=stereo.disparity_to_meters(disp,1000,100,2)
    assert depth[0]==pytest.approx(8.333333333)
    assert np.isnan(depth[1])


def test_sgbm_known_eight_pixel_translation():
    rng=np.random.default_rng(4)
    left=rng.integers(0,256,(64,192),np.uint8)
    right=np.zeros_like(left); right[:,:-8]=left[:,8:]
    disp=stereo.compute_sgbm(left,right,num_disparities=32,block_size=5)
    assert np.median(disp[8:-8,40:-20])==pytest.approx(8,abs=0.25)
