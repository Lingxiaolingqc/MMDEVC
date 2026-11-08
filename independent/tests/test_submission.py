import json
import zipfile

import cv2
import numpy as np
import pytest

from depth_project import submission
from depth_project.cache import save_predictions
from depth_project.render import render_comparison


@pytest.fixture
def run(tmp_path):
    root=tmp_path/'run';root.mkdir()
    values=np.ones((3,24,32),np.float32)
    render_comparison([0,1,2],{name:values for name in ['RGB','baseline','reference','fusion']},{name:(0,2) for name in ['RGB','baseline','reference','fusion']},12,root/'comparison.mp4')
    render_comparison([0,1,2],{name:values for name in ['baseline','fusion']},{name:(0,2) for name in ['baseline','fusion']},12,root/'background-blur.mp4')
    common={'frame_ids':[0,1,2],'source_sha256':'a'*64,'kind':'relative','output_hw':[24,32]}
    paths=[]
    for role,key in [('baseline','baseline-key'),('reference','reference-key'),('fusion','fusion-key')]:
        raw=root/f'raw_{role}.npz'
        extra={'parent_baseline_key':'baseline-key','configuration_sha256':'config-sha'} if role=='fusion' else {}
        save_predictions(raw,values,{**common,**extra,'key':key})
        paths.append(str(raw))
    summary={'processed_frames':3,'requested_frames':100,'fps':12,'compliant_100_frames':False,'source_sha256':'a'*64,'frame_ids':[0,1,2],'output_hw':[24,32],'baseline_cache_key':'baseline-key','configuration_sha256':'config-sha','video_reference':{'key':'reference-key'},'raw_prediction_files':paths}
    (root/'summary.json').write_text(json.dumps(summary))
    (root/'COMPLIANCE.txt').write_text('Only 3 genuine frames; this is a preview.')
    (root/'.env').write_text('DO_NOT_PACKAGE=private')
    return root


def test_short_video_cannot_be_claimed_submission_ready(run,tmp_path):
    with pytest.raises(ValueError,match='provisional|100'):
        submission.build_submission(run,tmp_path/'depth_results.zip')
    assert not (tmp_path/'depth_results.zip').exists()


def test_preview_contains_only_reviewed_outputs_and_checksum_inventory(run,tmp_path):
    target=tmp_path/'preview.zip'
    inventory=submission.build_submission(run,target,allow_short=True)
    assert inventory['submission_ready'] is False
    with zipfile.ZipFile(target) as archive:
        assert '.env' not in archive.namelist()
        assert 'COMPLIANCE.txt' in archive.namelist()
        assert 'manifest.json' in archive.namelist()
        assert 'raw_predictions/00_raw_baseline.npz' in archive.namelist()
        stored=json.loads(archive.read('manifest.json'))
        assert stored['processed_frames']==3
        assert all(len(row['sha256'])==64 for row in stored['files'])


def test_video_frame_count_mismatch_is_rejected(run,tmp_path):
    path=run/'summary.json';data=json.loads(path.read_text());data['processed_frames']=4;path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='frame'):
        submission.build_submission(run,tmp_path/'preview.zip',allow_short=True)


@pytest.mark.parametrize('defect,match',[('source','source'),('frame_ids','frame'),('shape','shape|size'),('kind','relative|kind'),('baseline_key','baseline|key'),('parent_key','baseline|parent'),('config_hash','config')])
def test_unrelated_or_incompatible_raw_evidence_is_rejected(run,tmp_path,defect,match):
    from depth_project.cache import load_predictions
    summary=json.loads((run/'summary.json').read_text())
    index=2 if defect in {'parent_key','config_hash'} else 0
    path=summary['raw_prediction_files'][index]
    values,metadata=load_predictions(path)
    if defect=='source':metadata['source_sha256']='b'*64
    elif defect=='frame_ids':metadata['frame_ids']=[100,101,102]
    elif defect=='shape':values=np.ones((3,1,1),np.float32);metadata['output_hw']=[1,1]
    elif defect=='kind':metadata['kind']='metric_m'
    elif defect=='baseline_key':metadata['key']='unrelated'
    elif defect=='parent_key':metadata['parent_baseline_key']='unrelated'
    elif defect=='config_hash':metadata['configuration_sha256']='unrelated'
    save_predictions(path,values,metadata)
    with pytest.raises(ValueError,match=match):
        submission.build_submission(run,tmp_path/'preview.zip',allow_short=True)


def test_wrong_rendered_dimensions_are_rejected(run,tmp_path):
    path=run/'comparison.mp4'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),12,(32,24))
    for _ in range(3):writer.write(np.zeros((24,32,3),np.uint8))
    writer.release()
    with pytest.raises(ValueError,match='size|dimension|shape'):
        submission.build_submission(run,tmp_path/'preview.zip',allow_short=True)
