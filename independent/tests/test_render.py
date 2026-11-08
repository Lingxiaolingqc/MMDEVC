import numpy as np
import pytest
from depth_project import render,application


def test_depth_value_keeps_same_color_across_frames():
    a=render.colorize(np.array([[0,5,10]],np.float32),(0,10))
    b=render.colorize(np.array([[5,6,7]],np.float32),(0,10))
    np.testing.assert_array_equal(a[0,1],b[0,0])


def test_whole_sequence_range_flat_map_and_saturation():
    lo,hi=render.compute_display_range(np.array([[[0,1]],[[9,10]]],np.float32))
    assert lo==pytest.approx(0.06)
    assert hi==pytest.approx(9.94)
    assert np.isfinite(render.colorize(np.ones((3,4)),(1,1))).all()
    np.testing.assert_array_equal(render.colorize(np.array([[-2]]),(0,10)),render.colorize(np.array([[0]]),(0,10)))


def test_background_blur_preserves_near_foreground():
    rgb=np.zeros((9,9,3),np.uint8);rgb[4,4]=255
    near=np.full((9,9),10,np.float32)
    far=np.zeros((9,9),np.float32)
    config={'display_range':[0,10],'focus_threshold':0.6,'transition':0.1,'blur_sigma':2}
    np.testing.assert_array_equal(application.background_blur(rgb,near,config),rgb)
    result=application.background_blur(rgb,far,config)
    assert result[4,4,0]<255
    assert result[4,3,0]>0


def test_comparison_video_preserves_real_frame_count_and_fps(tmp_path):
    from depth_project.video import inspect_video
    values=np.arange(3,dtype=np.float32)[:,None,None]*np.ones((3,24,32),np.float32)
    output=tmp_path/'comparison.mp4'
    render.render_comparison([0,1,2],{'Depth':values},{'Depth':(0,2)},12,output)
    info=inspect_video(output,required_frames=3)
    assert info.decoded_frames==3
    assert info.fps==12
