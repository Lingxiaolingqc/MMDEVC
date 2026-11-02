import unittest
import numpy as np
from p3_depth.application import background_blur


class ApplicationTests(unittest.TestCase):
    def test_near_unchanged_far_blurred_and_invalid_unchanged(self):
        grid=(np.indices((24,32)).sum(axis=0)%2*255).astype(np.uint8)
        rgb=np.repeat(grid[...,None],3,axis=2);valid=np.ones(grid.shape,bool)
        near,weight=background_blur(rgb,np.ones(grid.shape),valid,[0,1])
        np.testing.assert_array_equal(near,rgb);self.assertFalse(weight.any())
        far,weight=background_blur(rgb,np.zeros(grid.shape),valid,[0,1])
        self.assertLess(float(far.std()),float(rgb.std())/10)
        self.assertTrue((weight==1).all())
        missing,weight=background_blur(rgb,np.full(grid.shape,np.nan),valid,[0,1])
        np.testing.assert_array_equal(missing,rgb);self.assertFalse(weight.any())

    def test_fixed_limits_preserve_frame_to_frame_depth_change(self):
        rgb=np.zeros((12,16,3),np.uint8);valid=np.ones((12,16),bool)
        _,a=background_blur(rgb,np.full(valid.shape,.4),valid,[0,1])
        _,b=background_blur(rgb,np.full(valid.shape,.8),valid,[0,1])
        self.assertTrue((a==1).all());self.assertFalse(b.any())


if __name__=='__main__':unittest.main()
