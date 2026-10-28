"""Pinned official offline metric VDA; preserve temporal windows and stitching."""

import json
import sys
import numpy as np
from p3_depth.models.runtime import checked_source, file_hash


class VideoModel:
    def __init__(self, root, settings, device='cuda', precision='fp16', input_size=364, threads=8):
        import torch
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        if precision == 'fp16' and device != 'cuda':
            raise ValueError('Use CPU fp32 or CUDA fp16')
        if threads < 1 or input_size < 14:
            raise ValueError('Positive thread count and input_size >=14 required')
        torch.set_num_threads(threads)
        torch.manual_seed(13)
        sys.dont_write_bytecode = True
        asset = json.loads((root/'environments/model_assets.json').read_text(encoding='utf-8'))['vda_metric']
        source = (root/settings['third_party_root']/asset['source_directory']).resolve()
        checked_source(source, asset['source_commit'])
        sys.path.insert(0,str(source))
        from video_depth_anything.video_depth import VideoDepthAnything
        path = (root/settings['checkpoint_root']/'vda_metric'/asset['filename']).resolve()
        if file_hash(path) != asset['sha256']:
            raise ValueError('VDA checkpoint hash mismatch')
        self.model = VideoDepthAnything(encoder='vits', features=64, out_channels=[48,96,192,384], metric=True)
        self.model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
        self.model = self.model.to(device).eval()
        self.device, self.precision, self.input_size = device, precision, input_size
        self.shapes = set()
        # Upstream calls self.forward and get_intermediate_layers directly;
        # patch_embed is the first normal Module call receiving actual NCHW pixels.
        self.hook = self.model.pretrained.patch_embed.register_forward_pre_hook(
            lambda _, inputs: self.shapes.add(tuple(inputs[0].shape)))
        self.metadata = {'model': 'vda_metric', 'asset': asset, 'mode': 'official_offline', 'device': device,
            'precision': precision, 'input_size': input_size, 'threads': threads,
            'parameters': sum(p.numel() for p in self.model.parameters()), 'weight_bytes': path.stat().st_size,
            'gt_used_by_model': False, 'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
            'device_name': torch.cuda.get_device_name() if device == 'cuda' else 'CPU',
            'temporal_windows': 'official INFER_LEN=32, OVERLAP=10, padding/keyframes/interpolation unchanged',
            'alignment': 'official metric mode: scale=1 shift=0 between windows; no GT'}

    def __call__(self, frames, fps):
        if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8 or len(frames) == 0:
            raise ValueError('Expected nonempty uint8 NHWC RGB')
        depth, _ = self.model.infer_video_depth(frames, fps, input_size=self.input_size, device=self.device, fp32=self.precision=='fp32')
        if depth.shape != frames.shape[:3]:
            raise ValueError('VDA output frame count or dimensions differ from cache')
        if not self.shapes:
            raise RuntimeError('Actual encoder dimensions were not captured')
        self.metadata['encoder_shapes_nchw'] = [list(s) for s in sorted(self.shapes)]
        return depth.astype(np.float32,copy=False)
