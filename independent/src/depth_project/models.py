"""Thin adapters around official frozen pretrained models."""
from contextlib import nullcontext
from pathlib import Path
import sys

import cv2
import numpy as np

from .records import DepthPrediction, sha256_file


def validate_rgb(rgb):
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Input must be HWC RGB")
    if rgb.dtype != np.uint8:
        raise ValueError("Input RGB must have uint8 dtype")


class CallableImageAdapter:
    def __init__(self, predictor, *, kind, model_id, checkpoint_sha256, preprocessing_id, precision):
        self.predictor = predictor
        self.metadata = dict(kind=kind, model_id=model_id, checkpoint_sha256=checkpoint_sha256, preprocessing_id=preprocessing_id, precision=precision)

    def predict_rgb(self, rgb: np.ndarray) -> DepthPrediction:
        validate_rgb(rgb)
        prediction = DepthPrediction(self.predictor(rgb), **self.metadata)
        if prediction.values.shape != rgb.shape[:2]:
            raise ValueError("Prediction output size does not match input RGB")
        return prediction


def _source(root, name):
    source = Path(root) / "third_party" / name
    if not source.is_dir():
        raise FileNotFoundError(f"Official repository missing: {source}")
    if str(source.resolve()) not in sys.path:
        sys.path.insert(0, str(source.resolve()))
    return source


def load_image_adapter(name: str, root: Path, input_size: int = 518, precision: str = "fp16"):
    import torch

    if precision not in {"fp16", "fp32"}:
        raise ValueError("Precision must be fp16 or fp32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = lambda: torch.autocast("cuda", dtype=torch.float16) if device == "cuda" and precision == "fp16" else nullcontext()
    root = Path(root)
    if name == "dav2":
        _source(root, "Depth-Anything-V2")
        from depth_anything_v2.dpt import DepthAnythingV2

        weights = root / "checkpoints/depth_anything_v2_vits.pth"
        model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48,96,192,384])
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
        model = model.to(device).eval()
        def predict(rgb):
            with torch.inference_mode(), amp():
                return model.infer_image(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), input_size=input_size)
        adapter = CallableImageAdapter(predict, kind="relative", model_id="Depth-Anything-V2-Small", checkpoint_sha256=sha256_file(weights), preprocessing_id=f"official-infer_image-vits-{input_size}", precision=precision)
    elif name == "unidepth":
        _source(root, "UniDepth")
        from unidepth.models import UniDepthV2

        folder = root / "checkpoints/unidepth-v2-vits14"
        model = UniDepthV2.from_pretrained(str(folder.resolve())).to(device).eval()
        def predict(rgb):
            tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))).to(device)
            with torch.inference_mode():
                output = model.infer(tensor)
            return output["depth"][0,0].float().cpu().numpy()
        adapter = CallableImageAdapter(predict, kind="metric_m", model_id="UniDepthV2-ViT-S", checkpoint_sha256=sha256_file(folder / "model.safetensors"), preprocessing_id="official-infer-rgb-only-default-resolution", precision="official-fp16-autocast")
    else:
        raise ValueError(f"Unknown image model: {name}")
    adapter.model = model
    adapter.device = device
    return adapter


def infer_video_official(frames: np.ndarray, fps: float, root: Path, input_size: int = 518, precision: str = "fp16") -> np.ndarray:
    """Preserve the official offline overlap/keyframe/scale-alignment procedure."""
    import torch

    for rgb in frames:
        validate_rgb(rgb)
    _source(root, "Video-Depth-Anything")
    from video_depth_anything.video_depth import VideoDepthAnything

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VideoDepthAnything(encoder="vits", features=64, out_channels=[48,96,192,384], metric=False)
    weights = Path(root) / "checkpoints/video_depth_anything_vits.pth"
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    model = model.to(device).eval()
    output, returned_fps = model.infer_video_depth(frames, fps, input_size=input_size, device=device, fp32=precision == "fp32")
    if output.shape != frames.shape[:3] or returned_fps != fps or not np.isfinite(output).all():
        raise ValueError("Official video prediction shape/FPS/finite contract failed")
    return output.astype(np.float32)
