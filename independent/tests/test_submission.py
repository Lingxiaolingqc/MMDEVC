import json
import zipfile

import cv2
import numpy as np
import pytest

from depth_project import submission
from depth_project.cache import save_predictions


@pytest.fixture
def run(tmp_path):
    root=tmp_path/'run';root.mkdir()
    for name in ['comparison.mp4','background-blur.mp4']:
        writer=cv2.VideoWriter(str(root/name),cv2.VideoWriter_fourcc(*'mp4v'),12,(32,24))
        for _ in range(3):writer.write(np.zeros((24,32,3),np.uint8))
        writer.release()
    raw=root/'raw.npz';save_predictions(raw,np.ones((3,24,32)),{'frame_ids':[0,1,2]})
    summary={'processed_frames':3,'requested_frames':100,'fps':12,'compliant_100_frames':False,'raw_prediction_files':[str(raw)]}
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
        assert 'raw_predictions/00_raw.npz' in archive.namelist()
        stored=json.loads(archive.read('manifest.json'))
        assert stored['processed_frames']==3
        assert all(len(row['sha256'])==64 for row in stored['files'])


def test_video_frame_count_mismatch_is_rejected(run,tmp_path):
    path=run/'summary.json';data=json.loads(path.read_text());data['processed_frames']=4;path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='frame'):
        submission.build_submission(run,tmp_path/'preview.zip',allow_short=True)
