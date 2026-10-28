"""Analytical motion/occlusion checks and a real Farneback translation check."""
from pathlib import Path
import sys
import unittest
import cv2
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from p3_depth.temporal.flow import sample_previous,correspondence,estimate_pair
from p3_depth.temporal.fusion import fuse_step,fuse_sequence


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.p=yaml.safe_load((ROOT/'configs/fusion_provisional.yaml').read_text())['parameters']

    def test_backward_warp_direction_and_bilinear_missing_support(self):
        a=np.tile(np.arange(6,dtype=np.float32),(4,1))+1
        back=np.zeros((4,6,2),np.float32);back[...,0]=-1
        moved,valid=sample_previous(a,back)
        np.testing.assert_array_equal(moved[:,1:],a[:,:-1])
        self.assertFalse(valid[:,0].any())
        a[:,2]=np.nan;back[...,0]=-.5
        moved,valid=sample_previous(a,back)
        self.assertFalse(valid[:,2:4].any())
        self.assertTrue(np.isnan(moved[:,2:4]).all())

    def test_roundtrip_and_rgb_rejection_are_prediction_independent(self):
        rgb=np.zeros((4,8,3),np.uint8)
        f=np.zeros((4,8,2),np.float32);f[...,0]=1
        b=-f
        good=correspondence(rgb,rgb,f,b)
        self.assertTrue(good['common_mask'][:,1:].all())
        self.assertFalse(good['common_mask'][:,0].any())
        bad=correspondence(rgb,rgb,np.zeros_like(f),b)
        self.assertFalse(bad['common_mask'].any())
        photo=correspondence(rgb,np.full_like(rgb,255),f,b)
        self.assertFalse(photo['common_mask'].any())

    def test_flow_avoids_ema_ghost_and_does_not_fill_holes(self):
        previous=np.tile(np.array([1,1,1,4,4,4],np.float32),(4,1))
        current=np.tile(np.array([1,1,1,1,4,4],np.float32),(4,1))
        back=np.zeros((4,6,2),np.float32);back[...,0]=-1
        aligned,w=fuse_step(current,previous,'E2',self.p,backward=back)
        np.testing.assert_array_equal(aligned,current)
        ema,_=fuse_step(current,previous,'E1',self.p)
        self.assertGreater(ema[0,3],current[0,3])
        current[0,0]=np.nan
        out,w=fuse_step(current,previous,'E2',self.p,backward=back)
        self.assertTrue(np.isnan(out[0,0]));self.assertEqual(w[0,0],0)

    def test_filter_rejects_history_and_adaptive_responds_to_depth_change(self):
        current=np.ones((4,6),np.float32)
        previous=current*3
        flow=np.zeros((4,6,2),np.float32)
        fields=dict(forward=flow,backward=flow,photo_residual=np.zeros_like(current),fb_residual=np.zeros_like(current))
        fixed,_=fuse_step(current,previous,'E3',self.p,**fields)
        adaptive,_=fuse_step(current,previous,'E4',self.p,**fields)
        self.assertLess(np.mean(abs(adaptive-current)),.001)
        self.assertGreater(np.mean(abs(fixed-current)),1)
        fields['fb_residual'][:]=10
        rejected,w=fuse_step(current,previous,'E3',self.p,**fields)
        np.testing.assert_array_equal(rejected,current);self.assertFalse(w.any())

    def test_constant_sequence_and_first_frame_are_unchanged(self):
        d=np.full((5,4,6),2,np.float32);valid=np.ones_like(d,dtype=bool)
        out,w=fuse_sequence(d,valid,'E1',self.p)
        np.testing.assert_array_equal(out,d);self.assertFalse(w[0].any())

    def test_real_farneback_recovers_textured_translation(self):
        rng=np.random.default_rng(4)
        gray=cv2.GaussianBlur(rng.integers(0,256,(96,128),dtype=np.uint8),(3,3),.5)
        shifted=cv2.warpAffine(gray,np.float32([[1,0,3],[0,1,0]]),(128,96))
        rgb=np.repeat(gray[...,None],3,axis=2);next_rgb=np.repeat(shifted[...,None],3,axis=2)
        cfg=yaml.safe_load((ROOT/'configs/flow.yaml').read_text())
        forward,backward,fields=estimate_pair(rgb,next_rgb,cfg)
        self.assertAlmostEqual(float(np.median(forward[15:-15,15:-15,0])),3,delta=.15)
        self.assertAlmostEqual(float(np.median(backward[15:-15,15:-15,0])),-3,delta=.15)
        self.assertGreater(fields['common_mask'][15:-15,15:-15].mean(),.95)


if __name__=='__main__':
    unittest.main()
