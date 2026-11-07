"""Raw predictions, timestamps and model independence survive automatic previews."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from _common import checksum, write_json, write_csv
from _paths import output_path
from _media import nyu_gallery, video_preview, flow_preview


class AutomaticMediaTests(unittest.TestCase):
    def setUp(self):
        base=ROOT/'outputs/test-temp';base.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=base);self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def video_fixture(self):
        cache=self.root/'cache';cache.mkdir()
        rgb=np.zeros((3,48,64,3),np.uint8);rgb[:,:,:,0]=150
        np.save(cache/'frames.npy',rgb)
        write_csv(cache/'frames.csv',[{'timestamp_s':t} for t in [0,.1,.3]])
        cm={'status':'complete','scope':'fixture','frames':3,'requested_fps':10,
            'clip_id':'fixture','split':'dev','frames_sha256':checksum(cache/'frames.npy'),
            'index_sha256':checksum(cache/'frames.csv')}
        write_json(cache/'metadata.json',cm)
        return cache,cm

    def test_da_entry_generates_video_without_loading_vda(self):
        import _video_inference as runner
        cache,cm=self.video_fixture();run=self.root/'da'
        class FakeImageModel:
            def __init__(self,*args): self.metadata={'model':'dav2_relative'}
            def __call__(self,rgb): return np.linspace(-1,2,rgb.shape[0]*rgb.shape[1],dtype=np.float32).reshape(rgb.shape[:2])
        argv=['run_framewise.py','--cache',str(cache),'--output-dir',str(run),'--device','cpu','--precision','fp32']
        with patch.object(sys,'argv',argv),patch.object(runner,'ImageModel',FakeImageModel),patch.object(runner,'VideoModel',side_effect=AssertionError('DA must not load VDA')):
            runner.main(('dav2_relative',),'dav2_relative')
        raw=np.load(run/'depths.npy');self.assertEqual(raw.shape,(3,48,64));self.assertLess(float(raw.min()),0)
        self.assertTrue((run/'visualizations/comparison.mp4').is_file())
        self.assertTrue((run/'visualizations/preview.png').is_file())
        meta=json.loads((run/'visualizations/metadata.json').read_text())
        self.assertEqual([i['model'] for i in meta['inputs']],['dav2_relative'])
        before=checksum(run/'depths.npy');video_preview(cache,run)
        self.assertEqual(before,checksum(run/'depths.npy'))
        np.save(run/'depths.npy',raw+1)
        with self.assertRaisesRegex(ValueError,'Prediction artifact changed'):
            video_preview(cache,run)

    def test_all_nyu_images_have_previews_without_clipping_raw_values(self):
        run=self.root/'nyu';data=self.root/'data';data.mkdir();run.mkdir()
        for name in ['predictions','records']: (run/name).mkdir()
        rows=[]
        for sample in ['nyuv2_0003','nyuv2_0005']:
            rgb=data/f'{sample}.png';gt=data/f'{sample}.npy';pred=run/'predictions'/f'{sample}.npy'
            Image.new('RGB',(640,480),(80,120,180)).save(rgb)
            np.save(gt,np.full((480,640),2,np.float32));np.save(pred,np.full((480,640),12,np.float32))
            write_json(run/'records'/f'{sample}.json',{'rgb_sha256':checksum(rgb),'gt_sha256':checksum(gt),'prediction_sha256':checksum(pred)})
            rows.append({'sample_id':sample,'rgb_path':rgb.name,'depth_gt_path':gt.name})
        before=[checksum(p) for p in sorted((run/'predictions').iterdir())]
        nyu_gallery(run,rows,data,'unidepth')
        self.assertEqual(len(list((run/'visualizations').glob('*.png'))),2)
        self.assertTrue((run/'visualizations/index.html').is_file())
        self.assertEqual(before,[checksum(p) for p in sorted((run/'predictions').iterdir())])

    def test_flow_video_preserves_frame_count_and_irregular_timestamps(self):
        cache,cm=self.video_fixture();run=self.root/'flow';run.mkdir()
        arrays={'backward':np.zeros((2,48,64,2),np.float32),'common_mask':np.ones((2,48,64),bool)}
        for name,array in arrays.items():np.save(run/(name+'.npy'),array)
        write_json(run/'metadata.json',{'cache_metadata':cm,'artifacts_sha256':{name:checksum(run/(name+'.npy')) for name in arrays}})
        flow_preview(cache,run)
        meta=json.loads((run/'visualizations/metadata.json').read_text())
        self.assertEqual(meta['frames'],3);self.assertTrue(meta['raw_arrays_modified'] is False)
        flow_preview(cache,run)
        self.assertEqual(checksum(run/'visualizations/flow.mp4'),meta['video_sha256'])

    def test_defaults_are_part_specific_and_identifiers_cannot_escape(self):
        self.assertEqual(output_path('video_dav2',model='dav2_relative',split='dev',clip='tokyo'),ROOT/'outputs/part1_baseline/dav2_relative/dev/tokyo')
        self.assertEqual(output_path('nyu',model='unidepth',subset='test'),ROOT/'outputs/part2_models/unidepth/test')
        with self.assertRaises(ValueError):output_path('video_cache',clip='../escape')

    def test_registered_cache_rebuild_preserves_manifest_and_checks_destination(self):
        import prepare_video as runner
        raw=self.root/'data/videos/raw/tiny.mp4';raw.parent.mkdir(parents=True);raw.write_bytes(b'fixed source')
        output=self.root/'cache';manifest=self.root/'videos.csv'
        write_csv(manifest,[{'clip_id':'tiny','video_path':'raw/tiny.mp4','split':'dev',
            'scene_group':'room','scene_type':'camera_motion',
            'notes':f'source_sha256={checksum(raw)}; cache=cache'}])
        before=checksum(manifest)
        frames=np.zeros((2,48,64,3),np.uint8)
        decoded=(frames,[{'timestamp_s':0},{'timestamp_s':.1}],
                 {'original_frame_rate':True,'max_frames':2,'requested_fps':10})
        argv=['prepare_video.py','--video',str(raw),'--clip-id','tiny','--scene-group','room',
              '--split','dev','--scene-type','camera_motion','--original-fps','--source-kind','public_example',
              '--source-url','https://example.org/tiny.mp4','--expected-frames','2','--max-frames','2',
              '--manifest',str(manifest),'--registered','--output-dir',str(output)]
        with patch.object(runner,'ROOT',self.root),patch.object(runner,'repo_path',side_effect=lambda p:(self.root/p).resolve()):
            with patch.object(sys,'argv',argv),patch.object(runner,'decode_video',return_value=decoded) as decode:
                runner.main()
                decode.assert_called_once()
            self.assertEqual(checksum(manifest),before)
            self.assertEqual(json.loads((output/'metadata.json').read_text())['scope'],'public_example_project')
            argv[-1]=str(self.root/'wrong_destination')
            with patch.object(sys,'argv',argv),patch.object(runner,'decode_video',side_effect=AssertionError('must validate before decoding')):
                with self.assertRaisesRegex(ValueError,'differs from selected registration'):
                    runner.main()


if __name__=='__main__':unittest.main()
