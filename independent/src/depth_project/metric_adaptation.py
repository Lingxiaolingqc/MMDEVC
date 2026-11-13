"""Small internal adaptations that preserve UniDepthV2's metric geometry."""
import torch
from torch import nn
from torch.nn import functional as F


class LowRankQKV(nn.Module):
    """Add independent low-rank Q/V updates to a frozen fused QKV projection."""

    def __init__(self, base, rank=4):
        super().__init__()
        if base.out_features != 3 * base.in_features or rank < 1:
            raise ValueError("QKV adaptation requires a fused three-way linear projection")
        self.base = base.requires_grad_(False)
        channels = base.in_features
        self.q_down = nn.Linear(channels, rank, bias=False)
        self.q_up = nn.Linear(rank, channels, bias=False)
        self.v_down = nn.Linear(channels, rank, bias=False)
        self.v_up = nn.Linear(rank, channels, bias=False)
        nn.init.zeros_(self.q_up.weight)
        nn.init.zeros_(self.v_up.weight)

    def forward(self, values):
        query = self.q_up(self.q_down(values))
        value = self.v_up(self.v_down(values))
        update = torch.cat([query, torch.zeros_like(query), value], dim=-1)
        return self.base(values) + update


class ProjectionDepth(nn.Module):
    """The existing final UniDepth depth projections, operating on frozen inputs."""

    def __init__(self, input_channels=64):
        super().__init__()
        reduced = input_channels // 2
        self.to_depth_lr = nn.Conv2d(input_channels, reduced, 3, padding=1, padding_mode="reflect")
        self.to_depth_hr = nn.Sequential(
            nn.Conv2d(reduced, 32, 3, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, features, ray_z, output_hw=(480, 640)):
        latent = self.to_depth_lr(features)
        latent = F.interpolate(latent, size=ray_z.shape[-2:], mode="bilinear", align_corners=True)
        log_radius = self.to_depth_hr(latent)
        radius = torch.exp(log_radius.clamp(-8, 8) + 2)
        depth = radius * ray_z
        return F.interpolate(depth, size=output_hw, mode="bilinear", align_corners=False)

    def load_foundation(self, path):
        from safetensors import safe_open

        prefix = "pixel_decoder.depth_layer."
        with safe_open(str(path), framework="pt", device="cpu") as source:
            state = {key: source.get_tensor(prefix + key) for key in self.state_dict()}
        self.load_state_dict(state)
        return self


def configure_projections(model):
    model.requires_grad_(False)
    layer = model.pixel_decoder.depth_layer
    layer.to_depth_lr.requires_grad_(True)
    layer.to_depth_hr.requires_grad_(True)
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def configure_lora(model, rank):
    model.requires_grad_(False)
    names = [name for name, module in model.pixel_encoder.named_modules() if name.endswith("attn.qkv") and isinstance(module, nn.Linear)]
    if not names:
        raise ValueError("No fused encoder QKV projections found")
    for name in names:
        parent_name, attribute = name.rsplit(".", 1)
        parent = model.pixel_encoder.get_submodule(parent_name)
        base = getattr(parent, attribute)
        setattr(parent, attribute, LowRankQKV(base, rank).to(device=base.weight.device, dtype=base.weight.dtype))
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def align_depth_features(features, padded_hw, padding, output_hw):
    left, right, top, bottom = padding
    padded_height, padded_width = padded_hw
    height = padded_height - top - bottom
    width = padded_width - left - right
    target_height, target_width = output_hw
    y = top + (torch.arange(target_height, device=features.device, dtype=torch.float32) + 0.5) * height / target_height
    x = left + (torch.arange(target_width, device=features.device, dtype=torch.float32) + 0.5) * width / target_width
    yy, xx = torch.meshgrid(2 * y / padded_height - 1, 2 * x / padded_width - 1, indexing="ij")
    grid = torch.stack([xx, yy], dim=-1)[None].expand(len(features), -1, -1, -1)
    return F.grid_sample(features.float(), grid, mode="bilinear", padding_mode="border", align_corners=False)


def radius_to_z(radius, rays, output_hw, padding):
    points = radius * rays
    points = F.interpolate(points, size=output_hw, mode="bilinear", align_corners=False)
    left, right, top, bottom = padding
    height, width = output_hw
    return points[:, -1:, top:height - bottom, left:width - right]


def infer_differentiable(model, rgb):
    """Follow official infer preprocessing and geometry with autograd enabled."""
    from torchvision.transforms.v2.functional import normalize
    from unidepth.models.unidepthv2.unidepthv2 import get_paddings, get_resize_factor
    from unidepth.utils.constants import IMAGENET_DATASET_MEAN, IMAGENET_DATASET_STD

    if rgb.ndim == 3:
        rgb = rgb[None]
    rgb = rgb.to(model.device)
    height, width = rgb.shape[-2:]
    constraints = model.shape_constraints
    bounds = [constraints["pixels_min"], constraints["pixels_max"]]
    if hasattr(model, "resolution_level"):
        level = model.resolution_level
        if not 0 <= level < 10:
            raise ValueError("Resolution level must be between zero and nine")
        interval = (bounds[1] - bounds[0]) / 10
        bounds = [bounds[0] + level * interval, bounds[0] + (level + 1) * interval]
    padding, padded_hw = get_paddings((height, width), constraints["ratio_bounds"])
    _, resized_hw = get_resize_factor(padded_hw, bounds)
    image = normalize(rgb.float() / 255, IMAGENET_DATASET_MEAN, IMAGENET_DATASET_STD)
    left, right, top, bottom = padding
    image = F.pad(image, (left, right, top, bottom), value=0)
    image = F.interpolate(image, size=resized_hw, mode="bilinear", align_corners=False)
    _, outputs = model.encode_decode({"image": image, "camera": None}, image_metas=[])
    depth = F.interpolate(outputs["points"], size=padded_hw, mode="bilinear", align_corners=False)
    depth = depth[:, -1:, top:padded_hw[0] - bottom, left:padded_hw[1] - right]
    return depth
