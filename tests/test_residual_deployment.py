"""Live/test image formatting must match the recorded training inputs exactly."""
from pathlib import Path
import sys
import tempfile
import unittest
import cv2
import h5py
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from p3_depth.residual.deployment import image_inputs,needs_flip,load_refiner,predict_arrays
from p3_depth.residual.job_inputs import assemble_input_chunk,edges_at_resolution
from p3_depth.residual.model import candidate,Refiner
from p3_depth.residual.data import base_for


class DeploymentInputs(unittest.TestCase):
    def test_all_screening_and_combined_formats_match_training(self):
        rng=np.random.default_rng(5);hw=[12,16]
        rgb=rng.integers(0,256,(24,32,3),dtype=np.uint8)
        depth=rng.uniform(1.,5.,(24,32)).astype(np.float32)
        flip=depth*.96;confidence=rng.uniform(.1,1.,(24,32)).astype(np.float32)
        latent=rng.normal(size=(256,2,3)).astype(np.float16)
        calibration={'scale':{'scale':.95},'flip_scale':{'scale':.94}}
        edge_cal={'rgb_magnitude_normalizer':162.5,'depth_magnitude_normalizer':.2688,'rgb_high':128.,'depth_high':.06}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source.h5'
            with h5py.File(path,'w') as f:f.create_dataset('5/internal',data=latent)
            conf=cv2.resize(confidence,(16,12),interpolation=cv2.INTER_AREA)
            pack={'ids':[5],'cache_path':path,'original':depth[None],'flip':flip[None],
                  'rgb':(cv2.resize(rgb,(16,12),interpolation=cv2.INTER_AREA).transpose(2,0,1).astype(np.float32)/127.5-1)[None],
                  'confidence':(np.clip(np.log(np.maximum(conf,1e-6)),-5,5)/5)[None]}
            edge={k:v[None] for k,v in edges_at_resolution(rgb,depth,edge_cal,hw).items()}
            names=['R0','base_scale','base_flip','base_flip_scale','depth_only','flip','confidence','coordinates',
                   'frozen_features','rgb_sobel','rgb_canny','depth_sobel','depth_canny','flip_disagreement','joint_edges','legacy_bundle']
            configs=[candidate(name) for name in names]
            configs += [{**candidate('frozen_features'),'extra_inputs':['joint_edges'],'drop_rgb':True},
                        {**candidate('frozen_features'),'internal_mode':'zeros'}]
            for cfg in configs:
                base=base_for(pack,cfg['base'],calibration)
                expected,internal=assemble_input_chunk(pack,cfg,base,hw,edge)
                actual,actual_internal,actual_base=image_inputs(rgb,depth,cfg,hw,calibration,
                    flipped=flip,confidence=confidence,internal=latent,edge_calibration=edge_cal)
                np.testing.assert_array_equal(expected,actual)
                np.testing.assert_array_equal(base,actual_base)
                if internal is not None:np.testing.assert_array_equal(internal,actual_internal)
                else:self.assertIsNone(actual_internal)

    def test_flip_is_required_only_for_methods_that_use_it(self):
        rgb=np.zeros((16,24,3),np.uint8);depth=np.ones((16,24),np.float32)
        image_inputs(rgb,depth,candidate('R0'),[8,12],{})
        for name in ['flip','flip_disagreement','base_flip','legacy_bundle']:
            self.assertTrue(needs_flip(candidate(name)))
            with self.assertRaisesRegex(ValueError,'flip prediction'):image_inputs(rgb,depth,candidate(name),[8,12],{})

    def test_saved_models_reconstruct_all_architectures_for_inference(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            for architecture in ['cnn','mlp','unet','legacy_unet']:
                internal_channels=256 if architecture=='cnn' else 0
                original=Refiner(4,architecture,internal_channels)
                config={'candidate':{**candidate('R0'),'architecture':architecture}}
                path=Path(directory)/(architecture+'.pt')
                torch.save({'model':original.state_dict(),'config':config},path)
                restored,loaded_config=load_refiner(path,device='cpu')
                inputs=np.zeros((1,4,16,24),np.float32)
                latent=np.ones((1,256,2,3),np.float16) if internal_channels else None
                base=np.full((1,32,48),2.,np.float32)
                prediction=predict_arrays(restored,loaded_config,inputs,latent,base,device='cpu').numpy()
                np.testing.assert_array_equal(prediction,base)


if __name__=='__main__':unittest.main()
