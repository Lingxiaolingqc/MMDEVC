import math
import numpy as np
import pytest
from depth_project import metrics


def test_hand_computed_metric_depth():
    result=metrics.depth_metrics(np.array([1,2.4]), np.array([1,2]))
    assert result['abs_rel']==pytest.approx(0.1)
    assert result['rmse']==pytest.approx(math.sqrt(0.08))
    assert result['delta1']==1
    assert metrics.depth_metrics(np.array([1.25]),np.array([1.0]))['delta1']==0


def test_nyu_crop_and_strict_depth_endpoints():
    gt=np.zeros((480,640),np.float64)
    pred=np.ones_like(gt)
    gt[0,0]=1; pred[0,0]=100
    gt[45,41]=2; pred[45,41]=2.4
    gt[46,41]=0.001
    gt[47,41]=10
    gt[48,41]=np.nan
    result=metrics.evaluate_nyu(pred,gt)
    assert result['valid_count']==1
    assert result['abs_rel']==pytest.approx(0.2)
    assert result['rmse']==pytest.approx(0.4)


def test_prediction_clipping_and_resize():
    gt=np.ones((480,640),np.float64)
    result=metrics.evaluate_nyu(np.full((24,32),20.0),gt)
    assert result['rmse']==9
    assert result['valid_count']==426*560


def test_nonfinite_predictions_and_empty_mask_raise():
    with pytest.raises(ValueError,match='finite'):
        metrics.evaluate_nyu(np.full((480,640),np.nan),np.ones((480,640)))
    with pytest.raises(ValueError,match='valid'):
        metrics.evaluate_nyu(np.ones((480,640)),np.zeros((480,640)))


def test_equal_image_weighting_and_stereo_denominators():
    result=metrics.average_image_metrics([{'abs_rel':0.0,'rmse':0.0,'delta1':1.0,'valid_count':1000},{'abs_rel':1.0,'rmse':1.0,'delta1':0.0,'valid_count':1}])
    assert result['abs_rel']==0.5
    result=metrics.stereo_metrics(np.array([1,0,4,np.nan]),np.array([2,3,4,np.inf]))
    assert result['gt_valid_count']==3
    assert result['prediction_valid_count']==2
    assert result['coverage']==pytest.approx(2/3)
    assert result['mae_px']==0.5
