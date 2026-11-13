"""RGB-only inference with verified metric-depth LoRA checkpoints."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from safetensors.torch import load_file
import torch

from .lora_training import load_adapter_state
from .metric_adaptation import configure_lora
from .models import load_image_adapter, validate_rgb
from .records import DepthPrediction, sha256_file
from .refinement import blend_flip


class LoRADepthPredictor:
    def __init__(self, root, specification, foundation_sha256):
        self.flip_weight = float(specification.get("flip_weight", 0))
        if not math.isfinite(self.flip_weight) or not 0 <= self.flip_weight <= 1:
            raise ValueError("LoRA inference flip weight must be between zero and one")
        adapter = load_image_adapter("unidepth", root)
        if adapter.metadata["checkpoint_sha256"] != foundation_sha256:
            raise ValueError("LoRA inference foundation identity differs")
        if not specification["members"]:
            raise ValueError("LoRA inference requires a selected checkpoint")
        self.model, self.device = adapter.model, adapter.device
        configure_lora(self.model, specification["rank"])
        self.adapter_parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        names = [name for name, parameter in self.model.named_parameters() if parameter.requires_grad]
        self.members = []
        identities = []
        for member in specification["members"]:
            checkpoint = Path(root) / member["checkpoint"]
            if sha256_file(checkpoint) != member["sha256"]:
                raise ValueError("LoRA inference checkpoint identity differs")
            scale = float(member["scale"])
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("LoRA inference scale must be finite and positive")
            state = load_file(str(checkpoint), device=self.device)
            load_adapter_state(self.model, state)
            self.members.append(([state[name] for name in names], scale))
            identities.append({"sha256": member["sha256"], "scale": scale})
        identity = {"foundation_sha256": foundation_sha256, "rank": specification["rank"], "members": identities}
        if self.flip_weight:
            identity["flip_weight"] = self.flip_weight
        self.metadata = {"kind": "metric_m", "model_id": f"UniDepthV2-ViT-S-LoRA-r{specification['rank']}-{len(self.members)}member",
                         "checkpoint_sha256": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
                         "preprocessing_id": adapter.metadata["preprocessing_id"], "precision": adapter.metadata["precision"]}
        if self.flip_weight:
            self.metadata["model_id"] += f"-flip{self.flip_weight:g}"

    def predict_rgb(self, rgb):
        validate_rgb(rgb)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).to(self.device)
        predictions = []
        with torch.inference_mode():
            for weights, scale in self.members:
                for parameter, values in zip(self.adapter_parameters, weights):
                    parameter.copy_(values)
                depth = self.model.infer(tensor)["depth"][0, 0].float() * scale
                prediction = depth.cpu().numpy()
                if self.flip_weight:
                    output = self.model.infer(torch.flip(tensor, [-1]))
                    restored = torch.flip(output["depth"], [-1])[0, 0].float() * scale
                    prediction = blend_flip(prediction, restored.cpu().numpy(), self.flip_weight)
                predictions.append(prediction)
        values = np.mean(np.stack(predictions), axis=0, dtype=np.float64).astype(np.float32)
        return DepthPrediction(values, **self.metadata)
