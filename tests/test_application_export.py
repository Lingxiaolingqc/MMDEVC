"""Changing application settings preserves inputs and enables a matched comparison."""
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import av
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from _common import checksum,write_json,write_csv
from application import render_blur_demo as renderer


class ApplicationExportTests(unittest.TestCase):
    def setUp(self):
        parent=ROOT/'outputs/test-temp';parent.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=parent);self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.cache=self.root/'cache'
        self.cache.mkdir();self.times=[0.,.1,.3]
        grid=(np.indices((48,64)).sum(axis=0)%2*255).astype(np.uint8)
        frames=np.stack([np.repeat(grid[...,None],3,axis=2)]*3)
        np.save(self.cache/'frames.npy',frames)
        write_csv(self.cache/'frames.csv',[{'timestamp_s':t} for t in self.times])
        self.cm={'scope':'synthetic_application_check','clip_id':'fixture','split':'test',
                 'frames':3,'requested_fps':10,'frames_sha256':checksum(self.cache/'frames.npy'),
                 'index_sha256':checksum(self.cache/'frames.csv')}
        write_json(self.cache/'metadata.json',self.cm)
        depth=np.broadcast_to(np.linspace(0,1,64,dtype=np.float32),(3,48,64)).copy()
        self.limits=list(map(float,np.quantile(depth,[.01,.99])))
        self.runs=[]
        for name in ('dav2_relative','E4'):
            folder=self.root/name;folder.mkdir();self.runs.append(folder)
            np.save(folder/'depths.npy',depth);np.save(folder/'valid.npy',np.isfinite(depth))
            write_json(folder/'metadata.json',{'status':'complete','alignment':'none',
                'prediction_type':'inverse_depth_like','unit':'arbitrary','cache_metadata':self.cm,
                'model':{'model':name},'config_status':'frozen',
                'normalization_reference':{'limits':self.limits},
                'depths_sha256':checksum(folder/'depths.npy'),'valid_sha256':checksum(folder/'valid.npy')})

    def render(self,output,*extra):
        argv=['render_blur_demo.py','--cache',str(self.cache),'--baseline',str(self.runs[0]),
              '--improved',str(self.runs[1]),'--output-dir',str(output),*map(str,extra)]
        with patch.object(sys,'argv',argv),contextlib.redirect_stdout(io.StringIO()):
            renderer.main()

    def test_old_new_comparison_preserves_inputs_parameters_and_timestamps(self):
        previous=self.root/'original';adjusted=self.root/'adjusted'
        inputs={p:checksum(p) for folder in [self.cache,*self.runs] for p in folder.iterdir()}
        self.render(previous)
        old_files={p:checksum(p) for p in previous.iterdir()}
        self.render(adjusted,'--transition',.1,.3,'--reference-application',previous)
        self.assertEqual(old_files,{p:checksum(p) for p in old_files})
        self.assertEqual(inputs,{p:checksum(p) for p in inputs})
        meta=json.loads((adjusted/'metadata.json').read_text())
        self.assertEqual(meta['parameters']['transition_normalized'],[.1,.3])
        self.assertEqual(meta['parameters']['sigma_px'],5.)
        self.assertEqual(meta['settings_comparison']['previous_parameters']['transition_normalized'],[.45,.70])
        self.assertTrue(meta['settings_comparison']['same_predictions_and_fixed_scale'])
        for name,h in meta['files'].items():self.assertEqual(checksum(adjusted/name),h)
        for name in ('demo.mp4','settings_comparison.mp4'):
            with av.open(str(adjusted/name)) as container:
                times=[float(f.pts*f.time_base) for f in container.decode(video=0)]
            np.testing.assert_allclose(times,self.times,rtol=0,atol=2e-6)
        with (adjusted/'settings_comparison.csv').open() as handle:rows=list(csv.DictReader(handle))
        for row in rows:
            for method in ('DA','E4'):
                self.assertLess(float(row[method+'_adjusted_mean_blur_weight']),float(row[method+'_previous_mean_blur_weight']))
                self.assertGreater(float(row[method+'_mean_absolute_rgb_settings_change']),0)
        # Identical prediction arrays receive identical effect changes for both methods.
        self.assertEqual([r['DA_mean_absolute_rgb_settings_change'] for r in rows],
                         [r['E4_mean_absolute_rgb_settings_change'] for r in rows])

    def test_reference_from_different_predictions_rejected_before_creating_output(self):
        previous=self.root/'original';self.render(previous)
        path=previous/'metadata.json';meta=json.loads(path.read_text())
        meta['methods'][1]['metadata_sha256']='different-run';write_json(path,meta)
        target=self.root/'mismatch'
        with self.assertRaisesRegex(ValueError,'same cache, predictions'):
            self.render(target,'--transition',.1,.3,'--reference-application',previous)
        self.assertFalse(target.exists())

    def test_invalid_transition_rejected_before_creating_output(self):
        target=self.root/'invalid'
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            self.render(target,'--transition',.8,.2)
        self.assertFalse(target.exists())


if __name__=='__main__':unittest.main()
