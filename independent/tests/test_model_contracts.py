import numpy as np
import pytest

from depth_project import models


def test_rgb_input_and_depth_units_survive_adapter():
    received=[]
    def predict(rgb):
        received.append(rgb.copy())
        return rgb[:,:,0].astype(np.float64)
    adapter=models.CallableImageAdapter(predict, kind="relative", model_id="fixture", checkpoint_sha256="abc", preprocessing_id="identity", precision="fp32")
    image=np.array([[[255,0,0],[0,0,255]]],np.uint8)
    result=adapter.predict_rgb(image)
    np.testing.assert_array_equal(received[0],image)
    np.testing.assert_array_equal(result.values,[[255,0]])
    assert result.values.dtype==np.float32
    assert result.kind=="relative"


def test_invalid_input_and_output_are_rejected():
    adapter=models.CallableImageAdapter(lambda rgb: np.ones(rgb.shape[:2]),kind="metric_m",model_id="fixture",checkpoint_sha256="abc",preprocessing_id="identity",precision="fp32")
    with pytest.raises(ValueError,match="RGB"):
        adapter.predict_rgb(np.zeros((2,3),np.uint8))
    with pytest.raises(ValueError,match="uint8"):
        adapter.predict_rgb(np.zeros((2,3,3),np.float32))
    bad=models.CallableImageAdapter(lambda rgb: np.full(rgb.shape[:2],np.nan),kind="relative",model_id="fixture",checkpoint_sha256="abc",preprocessing_id="identity",precision="fp32")
    with pytest.raises(ValueError,match="finite"):
        bad.predict_rgb(np.zeros((2,3,3),np.uint8))
