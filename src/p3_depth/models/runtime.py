"""Small official model adapters. No GT is accepted by inference adapters."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def checked_source(path, commit):
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=path, text=True).strip()
    status = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=path, text=True)
    dirty = [line for line in status.splitlines() if not (
        line.startswith('?? ') and '__pycache__' in Path(line[3:]).parts and line.endswith('.pyc'))]
    if actual != commit or dirty:
        raise ValueError(f'Expected clean official source {path.name}@{commit}; actual={actual}, dirty={bool(dirty)}')


class ImageModel:
    def __init__(self, name, root, settings, device='cpu', input_size=434, precision='fp32', threads=8):
        import torch
        os.environ['WANDB_MODE'] = 'disabled'
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        sys.dont_write_bytecode = True
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable in this interpreter')
        if precision == 'fp16' and device != 'cuda':
            raise ValueError('FP16 requires CUDA')
        if threads < 1 or input_size < 14:
            raise ValueError('Positive thread count and input_size >=14 required')
        torch.set_num_threads(threads)
        torch.manual_seed(13)
        self.name, self.device, self.precision = name, device, precision
        self.input_size, self.encoder_shapes = input_size, set()
        upstream = (root / settings['third_party_root']).resolve()
        weights = (root / settings['checkpoint_root']).resolve()
        if name == 'unidepth':
            if device == 'cuda' and precision != 'fp16':
                raise ValueError('Official UniDepth CUDA infer uses FP16; explicitly request --precision fp16')
            from safetensors.torch import load_file
            source = upstream / 'UniDepth'
            commit = '8d8cfe4c7ee15297099983607febf0d4f32eb3d6'
            checked_source(source, commit)
            sys.path.insert(0, str(source))
            from unidepth.models import UniDepthV2
            path = weights / 'unidepth-v2-vits14/model.safetensors'
            expected = '93705cb3295dd7476b44911b8a55f5215bf74e8d5eccd27cecdb1b338270a648'
            config_path = path.parent / 'config.json'
            if file_hash(config_path) != 'ecc1f898690debe387d10329cca5d9d66a0a447ce83c3a2a7ae3673c7ce43cfe':
                raise ValueError('UniDepth config hash mismatch')
            config = json.loads(config_path.read_text(encoding='utf-8'))
            self.model = UniDepthV2(config)
            if file_hash(path) != expected:
                raise ValueError('UniDepth checkpoint hash mismatch')
            self.model.load_state_dict(load_file(str(path), device='cpu'), strict=True)
            self.model.resolution_level = 0
            encoder = self.model.pixel_encoder
            self.asset = {'source_commit': commit, 'filename': path.name, 'sha256': expected,
                          'prediction_type': 'metric_depth', 'unit': 'm', 'alignment': 'none',
                          'config_sha256': file_hash(config_path), 'resolution_level': 0}
        elif name in ('dav2_relative', 'dav2_metric'):
            self.asset = json.loads((root / 'environments/model_assets.json').read_text(encoding='utf-8'))[name]
            source = upstream / self.asset['source_directory']
            checked_source(source, self.asset['source_commit'])
            package_root = source / 'metric_depth' if name == 'dav2_metric' else source
            if 'depth_anything_v2.dpt' in sys.modules:
                loaded = Path(sys.modules['depth_anything_v2.dpt'].__file__).resolve()
                if not loaded.is_relative_to(package_root / 'depth_anything_v2'):
                    raise RuntimeError('Run relative and metric DA V2 in separate processes to avoid upstream package collision')
            sys.path.insert(0, str(package_root))
            from depth_anything_v2.dpt import DepthAnythingV2
            kwargs = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
            if name == 'dav2_metric':
                kwargs['max_depth'] = self.asset['max_depth_m']
            self.model = DepthAnythingV2(**kwargs)
            path = weights / name / self.asset['filename']
            if file_hash(path) != self.asset['sha256']:
                raise ValueError(f'{name} checkpoint hash mismatch')
            self.model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
            encoder = self.model.pretrained
        else:
            raise ValueError(f'Unsupported image model: {name}')
        self.model = self.model.to(device).eval()
        self.hook = encoder.register_forward_pre_hook(lambda _, inputs: self.encoder_shapes.add(tuple(inputs[0].shape)))
        self.metadata = {'model': name, 'asset': self.asset, 'device': device, 'requested_precision': precision,
                         'parameters': sum(p.numel() for p in self.model.parameters()), 'weight_bytes': path.stat().st_size,
                         'gt_used_by_model': False, 'camera_input': 'predicted intrinsics' if name == 'unidepth' else 'none',
                         'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
                         'device_name': torch.cuda.get_device_name() if device == 'cuda' else 'CPU',
                         'input_size': input_size if name != 'unidepth' else 'official resolution_level=0',
                         'threads': threads}

    def __call__(self, rgb):
        import torch
        import torch.nn.functional as F
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError('Expected uint8 RGB HWC')
        with torch.inference_mode():
            if self.name == 'unidepth':
                # Official infer owns its CUDA autocast and camera prediction.
                result = self.model.infer(torch.from_numpy(rgb.copy()).permute(2, 0, 1))
                depth = result['depth'][0, 0].float().cpu().numpy()
                self.metadata['actual_precision'] = 'official CUDA FP16 autocast' if self.device == 'cuda' else 'FP32'
            else:
                # Preserve official BGR input, interpolation, normalization and multiple-of-14 sizing.
                tensor, shape = self.model.image2tensor(rgb[..., ::-1].copy(), self.input_size)
                tensor = tensor.to(self.device)
                self.encoder_shapes.add(tuple(tensor.shape))
                with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.precision == 'fp16'):
                    output = self.model(tensor)
                    output = F.interpolate(output[:, None].float(), shape, mode='bilinear', align_corners=True)[0, 0]
                depth = output.float().cpu().numpy()
                self.metadata['actual_precision'] = self.precision
        if depth.shape != rgb.shape[:2]:
            raise ValueError('Model did not restore original image dimensions')
        self.metadata['encoder_shapes_nchw'] = [list(shape) for shape in sorted(self.encoder_shapes)]
        return depth.astype(np.float32, copy=True)
