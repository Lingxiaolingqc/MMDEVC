"""RGB-only inference for the development-selected LoRA/feature combination."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from safetensors.torch import load_file
import torch
from torch.nn import functional as F

from .lora_inference import LoRADepthPredictor
from .metric_training import scaled_features
from .models import load_image_adapter, validate_rgb
from .records import DepthPrediction, sha256_file
from .refinement import blend_flip
from .residual_refiner import ResidualRefiner, make_features


class FeatureDepthPredictor:
    def __init__(self, root, specification, foundation_sha256):
        adapter = load_image_adapter("unidepth", root)
        if adapter.metadata["checkpoint_sha256"] != foundation_sha256:
            raise ValueError("Feature inference foundation identity differs")
        self.model, self.device = adapter.model, adapter.device
        self.feature_hw = tuple(specification["feature_hw"])
        self.flip_weight = float(specification["flip_weight"])
        if not math.isfinite(self.flip_weight) or not 0 <= self.flip_weight <= 1 or not specification["members"]:
            raise ValueError("Feature inference requires valid blending and selected members")
        self.members = []
        for member in specification["members"]:
            path = Path(root) / member["checkpoint"]
            if sha256_file(path) != member["sha256"]:
                raise ValueError("Feature refiner checkpoint identity differs")
            scale = float(member["scale"])
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("Feature inference scale must be finite and positive")
            state = load_file(str(path), device=self.device)
            if any(not torch.isfinite(value).all() for value in state.values()):
                raise ValueError("Feature refiner checkpoint contains nonfinite values")
            refiner = ResidualRefiner(**specification["model"]).to(self.device).eval()
            refiner.load_state_dict(state)
            self.members.append((refiner, scale))
        identity = {"foundation": foundation_sha256, "specification": specification}
        self.metadata = {**adapter.metadata, "kind": "metric_m", "model_id": "UniDepthV2-S-feature-refiner",
                         "checkpoint_sha256": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()}

    def predict_members(self, rgb):
        validate_rgb(rgb)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).to(self.device)
        with torch.inference_mode():
            output = self.model.infer(tensor)
            original = output["depth"][0, 0].float().cpu().numpy()
            confidence = output["confidence"][0, 0].float().cpu().numpy()
            native = output["depth_features"].half().float()
            flipped = self.model.infer(torch.flip(tensor, [-1]))
            restored = torch.flip(flipped["depth"], [-1])[0, 0].float().cpu().numpy()
            base = blend_flip(original, restored, self.flip_weight)
            features = make_features(rgb, original, restored, confidence, base, self.feature_hw).astype(np.float16)[None]
            predictions = []
            for refiner, scale in self.members:
                values = scaled_features(features, base[None], scale).astype(np.float16).astype(np.float32)
                with torch.autocast(self.device, dtype=torch.float16, enabled=self.device == "cuda"):
                    residual = refiner(torch.from_numpy(values).to(self.device), native)
                residual = F.interpolate(residual.float(), size=rgb.shape[:2], mode="bilinear", align_corners=False)
                calibrated = torch.from_numpy(np.maximum(base * scale, 0.001)).to(self.device)
                predictions.append((calibrated * residual[0, 0].exp()).cpu().numpy())
        return predictions


class CombinedDepthPredictor:
    def __init__(self, root, specification, foundation_sha256):
        left = [member["seed"] for member in specification["lora"]["members"]]
        right = [member["seed"] for member in specification["refiner"]["members"]]
        if not left or left != right or len(set(left)) != len(left):
            raise ValueError("Combined inference requires identical unique paired seed identities")
        self.lora = LoRADepthPredictor(root, specification["lora"], foundation_sha256)
        self.refiner = FeatureDepthPredictor(root, specification["refiner"], foundation_sha256)
        self.model, self.device = self.lora.model, self.lora.device
        self.models = [self.lora.model, self.refiner.model]
        identity = [self.lora.metadata["checkpoint_sha256"], self.refiner.metadata["checkpoint_sha256"], left]
        self.metadata = {**self.lora.metadata, "model_id": "UniDepthV2-S-LoRA-feature-mean",
                         "checkpoint_sha256": hashlib.sha256(json.dumps(identity).encode()).hexdigest()}

    def predict_rgb(self, rgb):
        left, right = self.lora.predict_members(rgb), self.refiner.predict_members(rgb)
        if len(left) != len(right):
            raise ValueError("Combined inference member counts differ")
        pairs = [((a.astype(np.float64) + b) / 2).astype(np.float32) for a, b in zip(left, right)]
        values = np.mean(np.stack(pairs), axis=0, dtype=np.float64).astype(np.float32)
        return DepthPrediction(values, **self.metadata)


def load_research_predictor(root, specification, foundation_sha256):
    if specification.get("type", "lora") == "lora-feature-mean":
        return CombinedDepthPredictor(root, specification, foundation_sha256)
    if specification.get("type", "lora") != "lora":
        raise ValueError("Unknown research inference type")
    return LoRADepthPredictor(root, specification, foundation_sha256)
