"""Check fixed temporal denominators and missing/shifted depth boundary penalties."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from p3_depth.evaluation.temporal import evaluate_video
from p3_depth.evaluation.boundary import evaluate_boundary
from p3_depth.temporal.selection import validate_test_freeze


class VideoMetricTests(unittest.TestCase):
    def test_provisional_parameters_cannot_evaluate_final_video_test(self):
        with self.assertRaisesRegex(ValueError,'frozen'):
            validate_test_freeze(ROOT,{'status':'provisional','selection_evidence':None},{'split':'test'})

    def test_true_depth_change_and_timestamp_rate_are_not_aligned_away(self):
        d=np.stack([np.ones((4,6),np.float32),np.full((4,6),2,np.float32)])
        valid=np.ones_like(d,dtype=bool)
        flow=np.zeros((1,4,6,2),np.float32);region=np.ones((1,4,6),bool)
        s,rows=evaluate_video(d,valid,flow,region,[0,.5])
        self.assertEqual(s['temporal_l1_m_on_valid'],1)
        self.assertEqual(s['temporal_change_rate_m_per_s_on_valid'],2)
        self.assertEqual(s['agreement_5pct_full_reference'],0)
        self.assertAlmostEqual(s['median_abs_log_change'],np.log(2))

    def test_missing_predictions_do_not_reduce_fixed_denominator(self):
        d=np.ones((2,4,6),np.float32);v=np.ones_like(d,dtype=bool)
        v[1,:,:3]=False;d[1,:,:3]=np.nan
        region=np.ones((1,4,6),bool);copy=region.copy()
        s,_=evaluate_video(d,v,np.zeros((1,4,6,2),np.float32),region,[0,.1])
        self.assertEqual(s['reference_pixels'],24);self.assertEqual(s['pair_coverage'],.5)
        self.assertEqual(s['temporal_l1_m_on_valid'],0)
        self.assertEqual(s['capped_l1_m_full_reference'],.5)
        self.assertEqual(s['agreement_5pct_full_reference'],.5)
        np.testing.assert_array_equal(region,copy)

    def test_empty_reference_is_undefined_not_perfect(self):
        d=np.ones((2,4,6),np.float32)
        s,_=evaluate_video(d,np.ones_like(d,bool),np.zeros((1,4,6,2),np.float32),np.zeros((1,4,6),bool),[0,.1])
        self.assertIsNone(s['capped_l1_m_full_reference']);self.assertEqual(s['empty_reference_pairs'],1)

    def test_boundary_perfect_shifted_and_missing(self):
        gt=np.ones((20,30),np.float32);gt[:,15:]=2
        v=np.ones_like(gt,bool)
        perfect=evaluate_boundary(gt,gt,v,v)
        self.assertEqual(perfect['f1'],1);self.assertEqual(perfect['gt_to_prediction_capped_distance_px'],0)
        shifted=np.ones_like(gt);shifted[:,20:]=2
        shifted_scores=evaluate_boundary(shifted,gt,v,v)
        self.assertEqual(shifted_scores['f1'],0)
        self.assertAlmostEqual(shifted_scores['gt_to_prediction_capped_distance_px'],4.5)
        missing=evaluate_boundary(np.full_like(gt,np.nan),gt,v,np.zeros_like(v))
        self.assertEqual(missing['gt_edge_pixels'],perfect['gt_edge_pixels'])
        self.assertEqual(missing['recall'],0);self.assertEqual(missing['gt_to_prediction_capped_distance_px'],10)


if __name__=='__main__':
    unittest.main()
