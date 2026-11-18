"""Follow-up feature controls retain their stated information and normalization."""
from pathlib import Path
import sys
import tempfile
import unittest
import h5py
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from p3_depth.residual.job_inputs import assemble_input_chunk, edges_at_resolution
from p3_depth.residual.data import features_for


class FollowupInputs(unittest.TestCase):
    def test_combinations_preserve_reference_and_never_need_gt(self):
        pack={'original':np.full((2,16,24),2.,np.float32),'flip':np.full((2,16,24),3.,np.float32),
              'rgb':np.zeros((2,3,8,12),np.float32),'confidence':np.ones((2,8,12),np.float32)}
        reference,_=features_for(pack,{'inputs':'rgb_depth'},pack['original'],[8,12])
        value,_=assemble_input_chunk(pack,{'inputs':'rgb_depth','extra_inputs':['confidence','coordinates']},pack['original'],[8,12])
        np.testing.assert_array_equal(reference,value[:,:4])
        self.assertEqual(value.shape,(2,7,8,12))
        removed,_=assemble_input_chunk(pack,{'inputs':'rgb_depth','drop_rgb':True},pack['original'],[8,12])
        np.testing.assert_array_equal(removed,reference[:,3:])

    def test_zero_internal_control_preserves_shape_and_external_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'internal.h5'
            with h5py.File(path,'w') as data:data.create_dataset('4/internal',data=np.ones((256,2,3),np.float16))
            pack={'ids':[4],'cache_path':path,'original':np.ones((1,8,12),np.float32),
                  'rgb':np.zeros((1,3,8,12),np.float32)}
            ordinary,latent=assemble_input_chunk(pack,{'inputs':'frozen_features'},pack['original'],[8,12])
            control,zeros=assemble_input_chunk(pack,{'inputs':'frozen_features','internal_mode':'zeros'},pack['original'],[8,12])
            np.testing.assert_array_equal(ordinary,control)
            self.assertEqual(latent.shape,zeros.shape)
            self.assertTrue(np.all(zeros==0))

    def test_edge_features_retain_screening_quantization(self):
        rng=np.random.default_rng(4)
        rgb=rng.integers(0,256,(24,32,3),dtype=np.uint8)
        pred=np.full((24,32),2.,np.float32);pred[:,16:]=3.
        cal={'rgb_magnitude_normalizer':162.5,'depth_magnitude_normalizer':.2688,'rgb_high':128.,'depth_high':.06}
        values=edges_at_resolution(rgb,pred,cal,[12,16])
        for value in values.values():
            self.assertEqual(value.shape,(12,16))
            np.testing.assert_array_equal(value,value.astype(np.float16).astype(np.float32))
            self.assertTrue(np.isfinite(value).all())


if __name__=='__main__':unittest.main()
