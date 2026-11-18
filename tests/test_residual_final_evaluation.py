"""Synthetic end-to-end test metrics, saved arrays and interrupted-write recovery."""
from pathlib import Path
import json,sys,tempfile,unittest
from types import SimpleNamespace
from unittest.mock import patch
import h5py,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from p3_depth.residual.final_evaluation import evaluate_final
from p3_depth.residual.records import write_json,sha


class SyntheticSource:
    def __init__(self,path):self.path=path
    def __enter__(self):self.file=h5py.File(self.path,'r');return self
    def __exit__(self,*args):self.file.close()
    def load(self,index):
        depth=np.full((480,640),2.,np.float32);mask=np.zeros_like(depth,bool);mask[45:471,41:601]=True
        return SimpleNamespace(depth_m=depth,valid_gt=mask,scene='synthetic_'+str(index))


class FinalEvaluation(unittest.TestCase):
    def test_saved_arrays_match_known_scores_and_resume_partial_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'study';cache=out/'cache/test/predictions.h5';cache.parent.mkdir(parents=True)
            with h5py.File(cache,'w') as f:
                for i in [0,1]:
                    f.create_dataset(f'{i}/original',data=np.full((480,640),3.,np.float32))
                    f.create_dataset(f'{i}/flip',data=np.full((480,640),4.,np.float32))
            data=root/'source.h5'
            with h5py.File(data,'w') as f:f.create_dataset('images',data=np.zeros((2,3,640,480),np.uint8))
            model={'model':'fixture','asset':{},'requested_precision':'fp16','input_size':'fixture','camera_input':'none','gt_used_by_model':False}
            metadata={'status':'complete','cache_sha256':sha(cache),'config':{'ids':[0,1]},'model':model}
            write_json(cache.parent/'metadata.json',metadata);write_json(out/'cache/development/metadata.json',metadata)
            frozen={'test_ids':[0,1],'dataset_sha256':sha(data),'protocol':{},'test_history':'synthetic fixture'}
            study={'output':'study','data':'source.h5'}
            calibration={'scale':{'scale':2/3},'flip_scale':{'scale':2/3.25}}
            with (
                patch('p3_depth.residual.final_evaluation.load_selection',return_value=(frozen,'freeze')),
                patch('p3_depth.residual.final_evaluation.completed_refits',return_value=[]),
                patch('p3_depth.residual.final_evaluation.final_calibration',return_value=calibration),
                patch('p3_depth.residual.final_evaluation.NYUv2Labeled',SyntheticSource),
                patch('p3_depth.residual.final_evaluation.snapshot'),
                patch('p3_depth.residual.final_evaluation.save_preview'),
            ):
                evaluate_final(root,study)
                b0=json.loads((out/'final_test/B0/result.json').read_text())
                self.assertAlmostEqual(b0['metrics']['abs_rel'],.5)
                self.assertAlmostEqual(b0['metrics']['rmse_m'],1.)
                self.assertEqual(b0['metrics']['images'],2)
                for name in ['B1','B3']:
                    r=json.loads((out/f'final_test/{name}/result.json').read_text())
                    self.assertAlmostEqual(r['metrics']['abs_rel'],0.,places=6)
                # An incomplete saved dataset is recomputed, not accepted as zeros.
                (out/'final_test/B0/result.json').unlink()
                with h5py.File(out/'final_test/B0/predictions.h5','a') as f:
                    f['1'][:]=0.;f['1'].attrs['complete']=False
                evaluate_final(root,study)
                with h5py.File(out/'final_test/B0/predictions.h5','r') as f:
                    np.testing.assert_array_equal(f['1'][:],np.full((480,640),3.,np.float32))
                # A completed artifact cannot be silently replaced.
                with h5py.File(out/'final_test/B0/predictions.h5','a') as f:f['0'][0,0]=7.
                with self.assertRaisesRegex(ValueError,'Completed final evaluation changed'):evaluate_final(root,study)


if __name__=='__main__':unittest.main()
