import pytest
from depth_project import experiments
import cv2
import numpy as np


def test_missing_stereo_pairs_cannot_produce_a_summary(tmp_path):
    with pytest.raises(ValueError,match='15'):
        experiments.run_stereo(tmp_path/'empty',tmp_path/'output')
    assert not (tmp_path/'output'/'summary.json').exists()


def test_video_baseline_cache_keeps_real_frame_ids_and_refuses_overwrite(tmp_path):
    from depth_project.models import CallableImageAdapter
    from depth_project.cache import load_predictions
    path=tmp_path/'short.avi'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),12,(32,24))
    for value in [20,40,60]:
        writer.write(np.full((24,32,3),value,np.uint8))
    writer.release()
    adapter=CallableImageAdapter(lambda rgb:rgb[:,:,0].astype(np.float32),kind='relative',model_id='fixture',checkpoint_sha256='abc',preprocessing_id='identity',precision='fp32')
    output=tmp_path/'prediction.npz'
    metadata=experiments.run_video_model(path,output,'dav2',tmp_path,limit=100,max_res=32,adapter=adapter)
    values,stored=load_predictions(output)
    assert values.shape==(3,24,32)
    assert stored['frame_ids']==[0,1,2]
    assert stored['source_compliant'] is False
    with pytest.raises(FileExistsError):
        experiments.run_video_model(path,output,'dav2',tmp_path,adapter=adapter)


def test_fusion_replay_uses_cached_flow_and_resets_each_sequence():
    from depth_project.records import Frame
    frames=[Frame(i,i/10,np.zeros((24,32,3),np.uint8)) for i in range(3)]
    values=np.array([10,12,14],np.float32)[:,None,None]*np.ones((3,24,32),np.float32)
    flow=np.zeros((2,24,32,2),np.float32)
    config={'method':'flow','history_cap':0.5}
    actual,diagnostics=experiments.fuse_sequence(frames,values,flow,flow,config)
    np.testing.assert_array_equal(actual[:,10,10],[10,11,12.5])
    again,_=experiments.fuse_sequence(frames,values,flow,flow,config)
    np.testing.assert_array_equal(again,actual)
    assert len(diagnostics)==3


def test_nyu_run_rejects_incomplete_split_before_loading_model(tmp_path):
    from scipy.io import savemat
    folder=tmp_path/'data/nyu';folder.mkdir(parents=True)
    savemat(folder/'splits.mat',{'trainNdxs':[[1]],'testNdxs':[[2]]})
    with pytest.raises(ValueError,match='795|654'):
        experiments.run_nyu(tmp_path,tmp_path/'output')
