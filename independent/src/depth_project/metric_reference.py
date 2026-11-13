"""Replay the previously frozen metric refiner for paired comparisons."""
import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file
import torch

from .models import load_image_adapter, validate_rgb
from .records import DepthPrediction, sha256_file
from .refinement import refine_metric
from .residual_refiner import ResidualRefiner, make_features, apply_residual


class MetricRefinerReference:
    def __init__(self, root):
        root = Path(root)
        self.config = json.loads((root / "configs/metric-final.json").read_text())
        checkpoint = root / self.config["refiner_checkpoint"]
        if sha256_file(checkpoint) != self.config["refiner_sha256"]:
            raise ValueError("Previous refiner checkpoint identity differs")
        self.adapter = load_image_adapter("unidepth", root)
        if self.adapter.metadata["checkpoint_sha256"] != self.config["foundation_checkpoint_sha256"]:
            raise ValueError("Previous refiner foundation identity differs")
        self.device = self.adapter.device
        self.refiner = ResidualRefiner().to(self.device).eval()
        self.refiner.load_state_dict(load_file(str(checkpoint), device=self.device))
        self.metadata = {**self.adapter.metadata, "model_id": "UniDepthV2-ViT-S-residual-refiner",
                         "checkpoint_sha256": self.config["refiner_sha256"]}

    def predict_pair(self, rgb):
        validate_rgb(rgb)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).to(self.device)
        with torch.inference_mode():
            outputs = self.adapter.model.infer(tensor)
            original = outputs["depth"][0, 0].float().cpu().numpy()
            confidence = outputs["confidence"][0, 0].float().cpu().numpy()
            flipped = self.adapter.model.infer(torch.flip(tensor, [-1]))
            flip = torch.flip(flipped["depth"], [-1])[0, 0].float().cpu().numpy()
            base = refine_metric(rgb, original, flip, self.config["simple_config"])
            features = make_features(rgb, original, flip, confidence, base, tuple(self.config["feature_hw"]))
            with torch.autocast(self.device, dtype=torch.float16, enabled=self.device == "cuda"):
                residual = self.refiner(torch.from_numpy(features[None]).to(self.device))[0, 0].float().cpu().numpy()
            prediction = apply_residual(base, residual)
        return {"baseline_m": original, "previous_refiner_m": prediction}

    def predict_rgb(self, rgb):
        return DepthPrediction(self.predict_pair(rgb)["previous_refiner_m"], **self.metadata)
