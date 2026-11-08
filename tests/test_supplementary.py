"""Analytical controls and tiny saved-output round trips; no model inference."""
import contextlib
import csv
import io
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch
import uuid
from types import SimpleNamespace
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from _common import checksum,write_json,write_csv,read_yaml
from p3_depth.temporal.fusion import fuse_step,fuse_sequence,CONFIDENCE_TERMS
from p3_depth.temporal.selection import suite_artifacts
from p3_depth.evaluation.stereo import compare_lr_rejection
from p3_depth.evaluation.boundary import evaluate_boundary,boundary_masks
from part2_models.evaluate_nyuv2_boundaries import prepare_boundary_prediction,save_boundary_preview
from part3_improvement.run_fusion_suite import suite_scores
from part3_improvement import run_supplementary_ablations as runner
from part3_improvement.verify_supplementary_ablations import verify
from part1_baseline import analyze_stereo_lr as lr_runner
from part2_models import evaluate_nyuv2_boundaries as boundary_runner
from p3_depth.data.middlebury import Calibration
from p3_depth.models.stereo_sgbm import left_right_consistency


PARAMS={'history_weight':.7,'fb_relative':.01,'fb_absolute':.5,'photo_max':.2,
        'photo_sigma':.1,'flow_sigma_px':1.,'depth_span_sigma':.1,'edge_span_sigma':.1}


class SupplementMathTests(unittest.TestCase):
    def test_history_has_known_distinct_third_frame_all_variants(self):
        depths=np.broadcast_to(np.array([0,2,4],np.float32)[:,None,None],(3,4,6)).copy()
        fields={k:np.zeros((2,4,6,2) if k in ('forward','backward') else (2,4,6),np.float32)
                for k in ('forward','backward','photo_residual','fb_residual')}
        for variant in ('E1','E2','E3','E4'):
            # Disabling all soft terms gives a known constant-weight E4 control.
            a,wa=fuse_sequence(depths,np.isfinite(depths),variant,PARAMS,fields,representation='inverse_depth_like',confidence_terms=[])
            b,wb=fuse_sequence(depths,np.isfinite(depths),variant,PARAMS,fields,representation='inverse_depth_like',history_source='pairwise',confidence_terms=[])
            np.testing.assert_allclose(a[:,0,0],[0,.6,1.62],atol=2e-7)
            np.testing.assert_allclose(b[:,0,0],[0,.6,2.6],atol=2e-7)
            np.testing.assert_array_equal(a[:2],b[:2]);self.assertFalse(wa[0].any())

    def test_dropped_soft_term_does_not_disable_hard_gate(self):
        current=np.tile(np.linspace(1,2,6,dtype=np.float32),(4,1));flow=np.zeros((4,6,2),np.float32)
        fields={'backward':flow,'forward':flow,'photo_residual':np.full_like(current,.25),'fb_residual':np.zeros_like(current)}
        result,weight=fuse_step(current,current+.1,'E4',PARAMS,representation='inverse_depth_like',
                               confidence_terms=['flow','depth','edge'],**fields)
        np.testing.assert_array_equal(result,current);self.assertFalse(weight.any())
        fields['photo_residual'][:]=.1
        _,weight=fuse_step(current,current,'E4',PARAMS,representation='inverse_depth_like',confidence_terms=['photo'],**fields)
        np.testing.assert_allclose(weight,.7*np.exp(-1),rtol=1e-6)
        _,weight=fuse_step(current,current,'E4',PARAMS,representation='inverse_depth_like',confidence_terms=[],**fields)
        np.testing.assert_allclose(weight,.7)

    def test_invalid_switches_and_order(self):
        d=np.ones((3,4,6),np.float32)
        with self.assertRaises(ValueError):fuse_sequence(d,np.isfinite(d),'E1',PARAMS,history_source='typo')
        with self.assertRaises(ValueError):fuse_sequence(d,np.isfinite(d),'E1',PARAMS,confidence_terms=['edge','edge'])
        with self.assertRaises(ValueError):fuse_sequence(d,np.isfinite(d),'E1',PARAMS,confidence_terms=['unknown'])

    def test_lr_does_not_confuse_rejection_with_correction(self):
        before=np.array([[1,4,3,8]],np.float32);gt=np.array([[1,2,3,4]],np.float32)
        on=np.array([[True,False,False,True]]);off=np.ones_like(on)
        after=np.where(on,before,np.nan)
        scores=compare_lr_rejection(before,after,off,on,gt,off)
        self.assertEqual(scores['removed_gt_pixels'],2)
        self.assertEqual(scores['removed_bad_gt1px_fraction'],.5)
        self.assertEqual(scores['removed_within_1px_pixels'],1)
        after[0,0]=0
        with self.assertRaises(ValueError):compare_lr_rejection(before,after,off,on,gt,off)

    def test_course_boundary_clip_no_alignment_and_missing_rejection(self):
        gt=np.full((480,640),2,np.float32);gt[:,320:]=4
        raw=gt.copy();raw[:,500:]=15;raw[:,100:102]=-1
        original=raw.copy();asset={'prediction_type':'metric_depth','unit':'m','alignment':'none'}
        pred,valid,region=prepare_boundary_prediction(raw,np.isfinite(raw),gt,asset,'course_clipped')
        self.assertEqual(float(pred.max()),10);self.assertEqual(float(pred.min()),.001)
        np.testing.assert_array_equal(raw,original)
        perfect=evaluate_boundary(gt,gt,region,np.isfinite(gt))
        self.assertEqual(perfect['f1'],1)
        _,_,ge,pe=boundary_masks(gt,gt,region,np.isfinite(gt));np.testing.assert_array_equal(ge,pe)
        raw[100,100]=np.nan
        with self.assertRaisesRegex(ValueError,'Nonfinite'):prepare_boundary_prediction(raw,np.isfinite(raw),gt,asset,'course_clipped')
        with self.assertRaises(ValueError):prepare_boundary_prediction(original,valid,gt,{**asset,'alignment':'scale'},'course_clipped')


class SupplementExportTests(unittest.TestCase):
    def setUp(self):
        self.root=ROOT/'outputs/test-temp'/('supplement-'+uuid.uuid4().hex)
        self.root.mkdir(parents=True);self.addCleanup(self.cleanup)
        self.cache=self.root/'cache';self.base=self.root/'base';self.flow=self.root/'flow';self.ref=self.root/'reference'
        for path in (self.cache,self.base,self.flow,self.ref):path.mkdir()
        shape=(4,24,32);self.times=[0.,.1,.3,.5]
        self.depth=np.broadcast_to(np.linspace(0,5,32,dtype=np.float32),shape).copy()
        self.depth+=np.array([0,.3,-.1,.5],np.float32)[:,None,None]
        frames=np.repeat((self.depth*30).clip(0,255).astype(np.uint8)[...,None],3,axis=-1)
        np.save(self.cache/'frames.npy',frames);write_csv(self.cache/'frames.csv',[{'timestamp_s':t} for t in self.times])
        self.cm={'scope':'synthetic_supplement_check','clip_id':'fixture','split':'dev','scene_type':'camera_motion',
                 'requested_fps':10,'frames':4,'frames_sha256':checksum(self.cache/'frames.npy'),'index_sha256':checksum(self.cache/'frames.csv')}
        write_json(self.cache/'metadata.json',self.cm)
        self.limits=list(map(float,np.quantile(self.depth,[.01,.99])))
        self.save_prediction(self.base,self.depth,np.zeros_like(self.depth),'dav2_relative')
        bm=runner.read_json(self.base/'metadata.json');bm['inference_seconds']=.1;write_json(self.base/'metadata.json',bm)
        self.fields={k:np.zeros((3,24,32,2) if k in ('forward','backward') else (3,24,32),np.float32)
                     for k in ('forward','backward','photo_residual','fb_residual')}
        self.fields['photo_residual'][:]=.03;self.fields['common_mask']=np.ones((3,24,32),bool)
        for name,values in self.fields.items():np.save(self.flow/(name+'.npy'),values)
        write_json(self.flow/'metadata.json',{'status':'complete','cache_metadata':self.cm,'elapsed_seconds':.05,
            'artifacts_sha256':{k:checksum(self.flow/(k+'.npy')) for k in self.fields}})
        for variant in ('E0','E1','E2','E3','E4','E5'):
            pred,weight=fuse_sequence(self.depth,np.isfinite(self.depth),'E0' if variant=='E5' else variant,PARAMS,self.fields,representation='inverse_depth_like')
            folder=self.ref/variant;folder.mkdir()
            if variant!='E5':self.save_prediction(folder,pred,weight,variant)
            scores,pairs=suite_scores(pred,np.isfinite(pred),self.fields,self.times,True,False,self.limits)
            write_json(folder/'metrics.json',scores);write_csv(folder/'per_pair.csv',pairs)
        write_csv(self.ref/'comparison.csv',[{'variant':v} for v in ('E0','E1','E2','E3','E4','E5')])
        write_json(self.ref/'metadata.json',{'status':'complete','prediction_type':'inverse_depth_like','split':'dev',
            'config':{'status':'provisional','parameters':PARAMS},'baseline_normalization_limits':self.limits,
            **{key:checksum(path/'metadata.json') for path,key in [(self.cache,'cache_metadata_sha256'),(self.base,'base_metadata_sha256'),(self.flow,'flow_metadata_sha256')]}})
        write_json(self.ref/'verification.json',{'status':'verified','suite_metadata_sha256':checksum(self.ref/'metadata.json'),
            'artifacts_sha256':{k:checksum(self.ref/k) for k in suite_artifacts()}})

    def cleanup(self):
        root=self.root.resolve();parent=(ROOT/'outputs/test-temp').resolve()
        if not root.is_relative_to(parent) or not root.name.startswith('supplement-'):raise ValueError('Unsafe test cleanup')
        shutil.rmtree(root)

    def save_prediction(self,path,pred,weight,name):
        for key,array in [('depths',pred),('valid',np.isfinite(pred)),('history_weight',weight)]:np.save(path/(key+'.npy'),array)
        write_json(path/'metadata.json',{'status':'complete','model':{'model':name},'prediction_type':'inverse_depth_like',
            'unit':'arbitrary','alignment':'none','cache_metadata':self.cm,'fusion_seconds':.01,
            'normalization_reference':{'limits':self.limits},
            **{k+'_sha256':checksum(path/(k+'.npy')) for k in ('depths','valid','history_weight')}})

    def execute(self,experiment,extra=()):
        out=self.root/experiment
        argv=['runner','--cache',str(self.cache),'--baseline',str(self.base),'--flow',str(self.flow),
              '--reference-suite',str(self.ref),'--experiment',experiment,'--output-dir',str(out),*extra]
        with patch.object(sys,'argv',argv),contextlib.redirect_stdout(io.StringIO()):runner.main()
        return out

    def test_both_experiments_round_trip_and_leave_sources_unchanged(self):
        inputs={p:checksum(p) for root in (self.cache,self.base,self.flow,self.ref) for p in root.rglob('*') if p.is_file()}
        for group,count in [('history',9),('confidence',7)]:
            out=self.execute(group);result=verify(out)
            self.assertEqual(result['rows'],count);self.assertEqual(result['frames'],4)
            self.execute(group,['--visualize-only'])
        self.assertEqual(inputs,{p:checksum(p) for p in inputs})
        with self.assertRaises(FileExistsError):self.execute('history')
        target=self.root/'history/E1_pairwise/depths.npy'
        changed=np.load(target);changed[2,3,4]+=1;np.save(target,changed)
        with self.assertRaisesRegex(ValueError,'artifact changed'):verify(self.root/'history')

    def test_changed_source_rejected_before_output_creation(self):
        path=self.base/'depths.npy';changed=np.load(path);changed[1,1,1]+=1;np.save(path,changed)
        with self.assertRaisesRegex(ValueError,'hash mismatch'):self.execute('history')
        self.assertFalse((self.root/'history').exists())

    def test_lr_command_exports_saved_pair_analysis_and_figures(self):
        run=self.root/'stereo';folder=run/'Fixture';folder.mkdir(parents=True)
        for name in ('diagnostics','disparities','masks'):(folder/name).mkdir()
        before=np.full((24,32),2,np.float32);right=-before.copy();right[:,10:15]=-5
        gt=before.copy();gt[:12,12:17]=5
        valid,residual=left_right_consistency(before,right,1)
        for name,array in [('diagnostics/left_before_lr_px',before),('diagnostics/right_signed_px',right),
                           ('diagnostics/lr_error_px',residual),('disparities/Fixture',np.where(valid,before,np.nan)),('masks/Fixture',valid)]:
            np.save(folder/(name+'.npy'),array)
        (folder/'config_resolved.yaml').write_text('lr_max_diff_px: 1\nnum_disparities: 16\n',encoding='utf-8')
        manifest=self.root/'stereo.csv';manifest.write_text('synthetic manifest\n')
        write_json(folder/'metadata.json',{'status':'completed','manifest_sha256':checksum(manifest),'sources':[],
            'artifacts_sha256':{f.relative_to(folder).as_posix():checksum(f) for f in folder.rglob('*') if f.is_file()}})
        data=self.root/'data';data.mkdir();settings=self.root/'paths.yaml'
        settings.write_text('data_roots:\n  middlebury: '+data.as_posix()+'\n',encoding='utf-8')
        calibration=Calibration(np.diag([20.,20.,1.]),np.diag([20.,20.,1.]),100,0,32,24,16)
        sample=SimpleNamespace(paths=SimpleNamespace(sample_id='Fixture'),calibration=calibration,
            left_rgb=np.full((24,32,3),100,np.uint8),disparity_px=gt,depth_m=2/gt,
            valid_gt=np.ones((24,32),bool),nonoccluded=np.ones((24,32),bool))
        out=self.root/'lr-analysis';inputs={f:checksum(f) for f in folder.rglob('*') if f.is_file()}
        args=['lr','--run',str(run),'--manifest',str(manifest),'--paths-config',str(settings),'--output-dir',str(out)]
        with patch.object(sys,'argv',args),patch.object(lr_runner,'EXPECTED_SCENES',('Fixture',)),\
             patch.object(lr_runner,'select_sample',return_value=(sample.paths,'evaluation')),\
             patch.object(lr_runner,'load_sample',return_value=sample),contextlib.redirect_stdout(io.StringIO()):lr_runner.main()
        self.assertEqual(inputs,{f:checksum(f) for f in inputs})
        self.assertTrue((out/'visualizations/Fixture.png').is_file())
        with (out/'rejections.csv').open() as f:rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),2);self.assertGreater(int(rows[0]['removed_gt_pixels']),0)

    def test_boundary_command_exports_gallery_from_saved_predictions(self):
        run=self.root/'nyu';data=self.root/'nyu-data';data.mkdir();run.mkdir()
        for name in ('predictions','masks','records'):(run/name).mkdir()
        gt=np.full((480,640),2,np.float32);gt[:,320:]=4
        np.save(data/'gt.npy',gt);np.save(run/'predictions/fixture.npy',gt)
        np.save(run/'masks/fixture.npy',np.ones(gt.shape,bool));Image.fromarray(np.full((*gt.shape,3),100,np.uint8)).save(data/'rgb.png')
        write_json(run/'records/fixture.json',{'prediction_sha256':checksum(run/'predictions/fixture.npy'),
            'mask_sha256':checksum(run/'masks/fixture.npy'),'gt_sha256':checksum(data/'gt.npy'),'rgb_sha256':checksum(data/'rgb.png')})
        manifest=self.root/'nyu.csv';manifest.write_text('synthetic manifest\n')
        write_json(run/'metadata.json',{'status':'complete','scope':'synthetic_boundary_check',
            'model':{'model':'unidepth','gt_used_by_model':False,'asset':{'prediction_type':'metric_depth','unit':'m','alignment':'none'}},
            'identity':{'config':{'subset':'dev','limit':None},'manifest_sha256':checksum(manifest),'sample_ids':['fixture']}})
        settings=self.root/'nyu-paths.yaml';settings.write_text('data_roots:\n  nyuv2: '+data.as_posix()+'\n',encoding='utf-8')
        sample={'sample_id':'fixture','rgb_path':'rgb.png','depth_gt_path':'gt.npy'};out=self.root/'boundaries'
        args=['boundary','--run',str(run),'--paths-config',str(settings),'--output-dir',str(out)]
        with patch.object(sys,'argv',args),patch.object(boundary_runner,'select_nyu',return_value=(manifest,[sample])),contextlib.redirect_stdout(io.StringIO()):boundary_runner.main()
        self.assertEqual(runner.read_json(out/'summary.json')['pooled_f1'],1)
        self.assertTrue((out/'visualizations/index.html').is_file());self.assertTrue((out/'visualizations/fixture.png').is_file())


if __name__=='__main__':unittest.main()
