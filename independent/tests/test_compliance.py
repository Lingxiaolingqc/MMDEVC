import json

import cv2
import numpy as np

from depth_project.cache import save_predictions
from depth_project.evaluation import run_final_evaluation
from depth_project.experiments import run_video_model
from depth_project.models import CallableImageAdapter
from depth_project.records import sha256_file


def make_clip(path):
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),12,(32,24))
    for value in [50,60,70]:writer.write(np.full((24,32,3),value,np.uint8))
    writer.release()


def test_short_requested_limit_is_separate_from_course_compliance(tmp_path):
    source=tmp_path/'short.avi';make_clip(source)
    adapter=CallableImageAdapter(lambda rgb:rgb[:,:,0].astype(np.float32),kind='relative',model_id='fixture',checkpoint_sha256='abc',preprocessing_id='identity',precision='fp32')
    metadata=run_video_model(source,tmp_path/'prediction.npz','dav2',tmp_path,limit=3,max_res=32,adapter=adapter)
    assert metadata['source_compliant'] is False
    assert metadata['requested_limit_satisfied'] is True
    assert metadata['course_required_frames']==100


def test_final_evaluation_does_not_trust_legacy_limit_compliance(tmp_path):
    source=tmp_path/'short.avi';make_clip(source)
    values=np.array([5,6,7],np.float32)[:,None,None]*np.ones((3,24,32),np.float32)
    metadata={'key':'baseline','frame_ids':[0,1,2],'source_sha256':sha256_file(source),'source_decoded_frames':3,'source_required_frames':3,'source_compliant':True,'fps':12,'kind':'relative','output_hw':[24,32],'timing':{}}
    baseline=tmp_path/'baseline.npz';reference=tmp_path/'vda.npz'
    save_predictions(baseline,values,metadata)
    save_predictions(reference,values,{**metadata,'key':'reference'})
    config=tmp_path/'config.json'
    config.write_text(json.dumps({'method':'adaptive','history_cap':0.75,'selection':{'status':'frozen before rollercoaster fusion evaluation'}}))
    output=tmp_path/'result'
    result=run_final_evaluation(source,baseline,reference,config,output)
    assert result['processed_frames']==3
    assert result['compliant_100_frames'] is False
    warning=(output/'COMPLIANCE.txt').read_text()
    assert '3' in warning and '100' in warning
