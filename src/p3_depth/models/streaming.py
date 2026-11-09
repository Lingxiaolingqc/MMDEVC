"""Adapter for the pinned official relative VDA streaming implementation."""
import json
import sys
import numpy as np
from p3_depth.models.runtime import checked_source, file_hash


def reset_streaming_state(model):
    """Restore exactly the mutable state initialized by the pinned upstream class."""
    model.transform = None
    model.frame_id_list = []
    model.frame_cache_list = []
    model.id = -1
    for name in ("frame_height", "frame_width"):
        if hasattr(model, name):
            delattr(model, name)


class StreamingVideoModel:
    def __init__(self, root, settings, device="cuda", precision="fp16", input_size=364, threads=8):
        import torch
        if device != "cuda" or not torch.cuda.is_available() or precision != "fp16":
            raise ValueError("This matched experiment requires CUDA FP16")
        torch.set_num_threads(threads)
        torch.manual_seed(13)
        sys.dont_write_bytecode = True
        asset = json.loads((root / "environments/model_assets.json").read_text(encoding="utf-8"))["vda_relative"]
        source = (root / settings["third_party_root"] / asset["source_directory"]).resolve()
        checked_source(source, asset["source_commit"])
        sys.path.insert(0, str(source))
        from video_depth_anything.video_depth_stream import VideoDepthAnything
        weights = root / settings["checkpoint_root"] / "vda_relative" / asset["filename"]
        if file_hash(weights) != asset["sha256"]:
            raise ValueError("Streaming checkpoint mismatch")
        self.model = VideoDepthAnything(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
        self.model = self.model.to(device).eval()
        self.device, self.input_size = device, input_size
        self.shapes = set()
        self.hook = self.model.pretrained.patch_embed.register_forward_pre_hook(
            lambda _, inputs: self.shapes.add(tuple(inputs[0].shape)))
        self.metadata = {
            "model": "vda_relative_streaming", "mode": "official_streaming", "asset": asset,
            "device": device, "precision": precision, "input_size": input_size, "threads": threads,
            "device_name": torch.cuda.get_device_name(), "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda, "parameters": sum(p.numel() for p in self.model.parameters()),
            "gt_used_by_model": False, "alignment": "none; official cached temporal attention",
            "temporal_state": "first-frame initialization and official keyframe/history cache; one RGB frame per call"
        }

    def reset(self):
        reset_streaming_state(self.model)
        self.shapes.clear()

    def __call__(self, rgb):
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError("Expected one uint8 HWC RGB frame")
        value = self.model.infer_video_depth_one(
            rgb, input_size=self.input_size, device=self.device, fp32=False)
        if value.shape != rgb.shape[:2]:
            raise ValueError("Streaming output grid differs")
        self.metadata["encoder_shapes_nchw"] = [list(s) for s in sorted(self.shapes)]
        return value.astype(np.float32, copy=True)

