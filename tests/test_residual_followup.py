"""Small synthetic GPU integration: selected/final checkpoints and refit isolation."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from p3_depth.residual.followup import run_job
from p3_depth.residual.records import identity,sha,write_json


@unittest.skipUnless(torch.cuda.is_available(),'GPU integration requires CUDA')
class FollowupTraining(unittest.TestCase):
    def exercise(self,phase):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);study={'output':'study','data':'unused.mat'}
            candidate={'name':'fixture','base':'original','inputs':'rgb_depth','architecture':'cnn','output':'additive','loss':'abs_rel'}
            train={'batch_size':2,'max_epochs':2,'learning_rate':.001,'weight_decay':.001,'minimum_learning_rate':.0001,
                   'evaluate_every_epochs':1,'minimum_epochs':2,'patience_evaluations':5,'clip_gradient_norm':1.,
                   'gradient_loss_weight':.1,'boundary_loss_weight':2.}
            if phase=='data_size':train.update(max_steps=2,evaluate_every_steps=1)
            fits=[0,1,2,3] if phase=='refit' else [0,1];vals=[] if phase=='refit' else [2,3]
            job={'run_id':'fixture','phase':phase,'seed':17,'candidate':candidate,'training':train,
                 'fit_ids':fits,'validation_ids':vals,'feature_hw':[8,12],
                 'selection_point':'best_validation' if phase=='confirmation' else 'fixed_final'}
            pack={'ids':[0,1,2,3],'scenes':['a','a','b','c'],'metadata':{'cache_sha256':'synthetic'},
                  'original':np.full((4,480,640),2.,np.float32),'truth':np.full((4,480,640),1.8,np.float32),
                  'mask':np.zeros((4,480,640),bool),'band':np.zeros((4,480,640),bool)}
            pack['mask'][:,45:471,41:601]=True
            features=np.zeros((4,4,8,12),np.float32)
            def provenance(repo,out):write_json(out/'provenance.json',{'source_sha256':{}})
            with patch('p3_depth.residual.followup.read_job',return_value={'job':job,'job_sha256':identity(job)}),\
                 patch('p3_depth.residual.followup.fit_job_base',return_value=(pack['original'],{})),\
                 patch('p3_depth.residual.followup.prepare_job_features',return_value=(features,None,{'fixture':True})),\
                 patch('p3_depth.residual.followup.snapshot',side_effect=provenance),\
                 patch('p3_depth.residual.followup.preview'):
                result=run_job(root,study,None,shared_pack=pack)
            out=root/'study/followups/fixture'
            self.assertEqual(result['status'],'complete')
            self.assertFalse((out/'resume.pt').exists())
            for name,digest in result['checkpoint_sha256'].items():self.assertEqual(sha(out/name),digest)
            self.assertEqual(len(json.loads((out/'history.json').read_text())),2)
            if phase=='refit':self.assertIsNone(result['validation'])
            else:self.assertEqual(result['validation']['images'],2)
            self.assertEqual(result['selected_checkpoint'],'best.pt' if phase=='confirmation' else 'final.pt')

    def test_best_selection_retains_final_weights(self):self.exercise('confirmation')
    def test_data_size_scores_fixed_final_not_best(self):self.exercise('data_size')
    def test_all_development_refit_never_selects_on_validation(self):self.exercise('refit')


if __name__=='__main__':unittest.main()
