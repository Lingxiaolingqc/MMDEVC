"""A freeze must constrain the actual final model, not just exist as a file."""
from pathlib import Path
import sys,copy,unittest,json,tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from p3_depth.residual.selection import check_refit_spec,freeze_selection,load_selection,NUMERICAL_SOURCES,SHARED_NUMERICAL_SOURCES
from p3_depth.residual.jobs import build_job,save_job,read_job
from p3_depth.residual.model import candidate
from p3_depth.residual.records import write_json


class FreezeConstraints(unittest.TestCase):
    def setUp(self):
        self.job={'run_id':'final_R0_seed1','seed':1,'candidate':{'inputs':'rgb_depth'},'training':{'max_epochs':24},
                  'feature_hw':[240,320],'fit_ids':[0,1,2],'validation_ids':[],'selection_point':'fixed_final'}
        self.frozen={'refits':[copy.deepcopy(self.job)],'development_ids':[0,1,2]}

    def test_frozen_spec_accepted(self):check_refit_spec(self.job,self.frozen)

    def test_changed_method_duration_or_resolution_rejected(self):
        for key,value in [('candidate',{'inputs':'frozen_features'}),('training',{'max_epochs':80}),('feature_hw',[480,640]),('seed',2)]:
            changed={**self.job,key:value}
            with self.assertRaises(ValueError):check_refit_spec(changed,self.frozen)

    def test_validation_selection_or_test_data_rejected(self):
        for change in [{'validation_ids':[3]},{'fit_ids':[0,1,3]},{'selection_point':'best_validation'}]:
            with self.assertRaises(ValueError):check_refit_spec({**self.job,**change},self.frozen)

    def test_saved_refit_requires_real_freeze_and_unchanged_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            split={'fit_ids':[0,1],'validation_ids':[2,3],'test_ids':[4,5],'dataset_sha256':'synthetic'}
            study={'output':'study','split':'split.json','feature_hw':[240,320],'confirmation_seeds':[1,2,3],
                   'training':{'max_epochs':80,'batch_size':8},'protocol':{'test_history':'fixture'}}
            write_json(root/'split.json',split)
            for relative in [*[f'src/p3_depth/residual/{n}' for n in NUMERICAL_SOURCES],*SHARED_NUMERICAL_SOURCES]:
                path=root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fixture source',encoding='utf-8')
            write_json(root/'study/runs/result.json',{'status':'complete','smoke':False,'validation':{'images':218}})
            refits=[{'run_id':f'final_R0_seed{seed}','method':'R0','seed':seed,'candidate':candidate('R0'),
                     'feature_hw':[240,320],'training':{'max_epochs':24,'batch_size':8}} for seed in [1,2,3]]
            proposal={'refits':refits,'development_evidence':['study/runs/result.json'],'rationale':'fixture','duration_rule':'fixed24'}
            frozen,digest=freeze_selection(root,study,proposal)
            row=refits[0]
            job=build_job(study,split,'R0','refit',1,row['run_id'],fitting_ids=[0,1,2,3],
                          training_overrides=row['training'],frozen_selection_sha256=digest)
            path=save_job(root,study,job)
            self.assertEqual(read_job(root,study,path)['job'],job)
            altered=copy.deepcopy(job);altered['training']['max_epochs']=80
            with self.assertRaisesRegex(ValueError,'frozen training'):save_job(root,study,altered)
            (root/'src/p3_depth/residual/model.py').write_text('changed',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Numerical implementation'):load_selection(root,study)


if __name__=='__main__':unittest.main()
