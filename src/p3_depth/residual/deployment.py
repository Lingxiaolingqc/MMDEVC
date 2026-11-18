"""Single-image residual inputs and inference: no GT or evaluation masks accepted."""
import cv2
import numpy as np
import torch
from .data import base_for
from .job_inputs import assemble_input_chunk,edges_at_resolution
from .model import Refiner,corrected_depth


def needs_flip(config):
    return config['base'] in ('flip','flip_scale') or config['inputs'] in ('flip','flip_disagreement','legacy') or bool(set(config.get('extra_inputs',[])) & {'flip','flip_disagreement'})


def needs_edges(config):
    names=[config['inputs'],*config.get('extra_inputs',[])]
    return any(name in ('rgb_sobel','rgb_canny','depth_sobel','depth_canny','joint_edges') for name in names)


def image_inputs(rgb,original,config,hw,calibration,*,flipped=None,confidence=None,internal=None,edge_calibration=None):
    """Reproduce training preprocessing from observable inference fields only.

    Flipped prediction must already be restored to the original orientation.
    Internal maps are rounded to the same float16 storage used during training.
    """
    original=np.asarray(original,np.float32)
    if rgb.shape[:2]!=original.shape or original.ndim!=2:
        raise ValueError('RGB/depth shape mismatch')
    if not np.isfinite(original).all() or (original<=0).any():raise ValueError('Invalid foundation prediction')
    if needs_flip(config) and flipped is None:raise ValueError('This method requires a restored flip prediction')
    height,width=hw
    pack={'original':original[None],
          'flip':np.asarray(flipped,np.float32)[None] if flipped is not None else original[None],
          'rgb':(cv2.resize(rgb,(width,height),interpolation=cv2.INTER_AREA).transpose(2,0,1).astype(np.float32)/127.5-1)[None]}
    if config['inputs'] in ('confidence','legacy') or 'confidence' in config.get('extra_inputs',[]):
        if confidence is None:raise ValueError('This method requires official confidence')
        value=cv2.resize(np.asarray(confidence,np.float32),(width,height),interpolation=cv2.INTER_AREA)
        pack['confidence']=(np.clip(np.log(np.maximum(value,1e-6)),-5,5)/5)[None]
    base=base_for(pack,config['base'],calibration)
    edge=None
    if needs_edges(config):
        if edge_calibration is None:raise ValueError('Missing training-side edge calibration')
        edge={key:value[None] for key,value in edges_at_resolution(rgb,original,edge_calibration,hw).items()}
    construction={**config}
    if config['inputs']=='frozen_features':
        if internal is None:raise ValueError('This method requires frozen internal features')
        # Avoid file access in live inference; all observable arrays are supplied.
        construction.update(inputs='rgb_depth',internal_mode='true')
    features,_=assemble_input_chunk(pack,construction,base,hw,edge)
    latent=None
    if config['inputs']=='frozen_features':
        latent=np.asarray(internal,np.float16)[None]
        if config.get('internal_mode','true')=='zeros':latent=np.zeros_like(latent)
        if not np.isfinite(latent).all():raise ValueError('Invalid internal feature')
    if not np.isfinite(features).all():raise ValueError('Invalid refiner input')
    return features,latent,base


def load_refiner(path,device='cuda'):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    config=saved['config']
    state=saved['model']
    latent_channels=state['projection.weight'].shape[1] if 'projection.weight' in state else 0
    candidate=config['candidate']
    first='hidden.0.weight' if candidate['architecture'] in ('cnn','mlp') else 'encoders.0.0.weight'
    channels=state[first].shape[1]-(16 if latent_channels else 0)
    model=Refiner(channels,candidate['architecture'],latent_channels).to(device).eval()
    model.load_state_dict(state)
    return model,config


@torch.inference_mode()
def predict_arrays(model,config,features,internal,base,device='cuda'):
    def tensor(value):return torch.from_numpy(np.ascontiguousarray(value)).to(device)
    inputs=tensor(features);latent=tensor(internal).float() if internal is not None else None
    bases=tensor(base)[:,None]
    with torch.autocast(device_type=device.split(':')[0],dtype=torch.float16,enabled=device.startswith('cuda')):
        residual=model(inputs,latent)
    prediction=corrected_depth(bases,residual,config['candidate']['output'])
    if not bool(torch.isfinite(prediction).all()):raise ValueError('Nonfinite refined depth')
    return prediction[:,0]
