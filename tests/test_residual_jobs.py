"""Prevent test/validation leakage and post-hoc mutation in follow-up experiments."""
from pathlib import Path
import sys
import copy
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from p3_depth.residual.jobs import nested_scene_subsets, build_job, validate_job


class JobTests(unittest.TestCase):
    def setUp(self):
        self.split = {'fit_ids':list(range(20)), 'validation_ids':list(range(20,25)), 'test_ids':list(range(25,30))}
        self.study = {'feature_hw':[240,320], 'training':{'max_epochs':80,'batch_size':8},
                      'protocol':{'test_history':'previously inspected'}}

    def test_nested_whole_scenes_and_repeatability(self):
        scenes = {i:'scene'+str(i//3) for i in range(20)}
        result=nested_scene_subsets(self.split['fit_ids'],scenes)
        self.assertEqual(result,nested_scene_subsets(self.split['fit_ids'],scenes))
        previous=set()
        for row in result['subsets']:
            ids=set(row['fit_ids'])
            self.assertTrue(previous.issubset(ids))
            for scene in row['scenes']:
                self.assertTrue({i for i,s in scenes.items() if s==scene}.issubset(ids))
            previous=ids
        self.assertEqual(previous,set(self.split['fit_ids']))

    def test_test_images_rejected(self):
        job=build_job(self.study,self.split,'R0','confirmation',1,'example')
        job['fit_ids'][0]=25
        with self.assertRaisesRegex(ValueError,'Test IDs'):
            validate_job(job,self.split)

    def test_validation_cannot_become_fitting_before_refit(self):
        job=build_job(self.study,self.split,'R0','combination',1,'example')
        job['fit_ids'][0]=20
        with self.assertRaises(ValueError):validate_job(job,self.split)

    def test_refit_has_no_validation_selection(self):
        ids=self.split['fit_ids']+self.split['validation_ids']
        job=build_job(self.study,self.split,'R0','refit',1,'final',fitting_ids=ids,frozen_selection_sha256='frozen')
        self.assertEqual(job['validation_ids'],[])
        job['selection_point']='best_validation'
        with self.assertRaises(ValueError):validate_job(job,self.split)

    def test_data_size_requires_fixed_step_budget(self):
        with self.assertRaises(ValueError):
            build_job(self.study,self.split,'R0','data_size',1,'small',fitting_ids=[0,1,2])
        job=build_job(self.study,self.split,'R0','data_size',1,'small',fitting_ids=[0,1,2],training_overrides={'max_steps':2000})
        self.assertEqual(job['selection_point'],'fixed_final')


if __name__=='__main__':unittest.main()
