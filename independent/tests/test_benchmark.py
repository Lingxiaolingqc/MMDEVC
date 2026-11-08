import numpy as np
from depth_project import benchmark
from depth_project.models import CallableImageAdapter
from depth_project.records import Frame


def test_pipeline_benchmark_recomputes_predictions_and_resets_history():
    calls=[]
    def predict(rgb):
        calls.append(1)
        return np.full(rgb.shape[:2],10,np.float32)
    model=CallableImageAdapter(predict,kind='relative',model_id='fixture',checkpoint_sha256='abc',preprocessing_id='identity',precision='fp32')
    frames=[Frame(i,i/12,np.zeros((24,32,3),np.uint8)) for i in range(3)]
    result=benchmark.benchmark_frames(frames,model,{'method':'baseline'},repeats=3)
    assert len(calls)==11  # two warmups plus all three frames in all three trials
    assert len(result['trials'])==3
    assert all(row['frame_count']==3 for row in result['trials'])
    assert all(row['flow_s']==0 for row in result['trials'])
