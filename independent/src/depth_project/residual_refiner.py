"""Small residual network that adapts frozen metric predictions using RGB."""
import cv2
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def make_features(rgb, original, flip, confidence, base, output_hw=(240, 320)):
    height, width = output_hw
    resize = lambda values: cv2.resize(values, (width, height), interpolation=cv2.INTER_AREA)
    image = resize(rgb).astype(np.float32) / 127.5 - 1
    maps = [np.log(np.maximum(resize(values), .001)) / 2 for values in [original, flip, base]]
    certainty = np.clip(np.log(np.maximum(resize(confidence), 1e-6)), -5, 5) / 5
    yy, xx = np.meshgrid(np.linspace(-1, 1, height), np.linspace(-1, 1, width), indexing='ij')
    return np.concatenate([image.transpose(2, 0, 1), np.stack([*maps, certainty, xx, yy])]).astype(np.float32)


def apply_residual(base, log_residual):
    residual = cv2.resize(log_residual, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.asarray(base, dtype=np.float32) * np.exp(residual)


def conv_block(inputs, outputs):
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 3, padding=1), nn.GroupNorm(4, outputs), nn.GELU(),
        nn.Conv2d(outputs, outputs, 3, padding=1), nn.GroupNorm(4, outputs), nn.GELU(),
    )


class ResidualRefiner(nn.Module):
    def __init__(self, widths=(16, 32, 64, 96), dropout=0.0, feature_channels=0, gated=False, residual_bound=0.5):
        super().__init__()
        if len(widths) != 4 or any(width < 4 or width % 4 for width in widths):
            raise ValueError("Four refiner widths must be positive multiples of four")
        if not math.isfinite(residual_bound) or residual_bound <= 0:
            raise ValueError("Residual bound must be finite and positive")
        self.residual_bound = residual_bound
        self.feature_projection = nn.Conv2d(feature_channels, 16, 1) if feature_channels else None
        self.encoder0 = conv_block(25 if feature_channels else 9, widths[0])
        self.encoder1 = conv_block(widths[0], widths[1])
        self.encoder2 = conv_block(widths[1], widths[2])
        self.encoder3 = conv_block(widths[2], widths[3])
        self.bottleneck_dropout = nn.Dropout2d(dropout)
        self.decoder2 = conv_block(widths[3] + widths[2], widths[2])
        self.decoder1 = conv_block(widths[2] + widths[1], widths[1])
        self.decoder0 = conv_block(widths[1] + widths[0], widths[0])
        self.output = nn.Conv2d(widths[0], 1, 1)
        self.gate = nn.Conv2d(widths[0], 1, 1) if gated else None
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features, depth_features=None):
        if self.feature_projection is not None:
            if depth_features is None:
                raise ValueError("This refiner requires frozen depth features")
            projected = self.feature_projection(depth_features)
            projected = F.interpolate(projected, size=features.shape[-2:], mode='bilinear', align_corners=False)
            features = torch.cat([features, projected], dim=1)
        level0 = self.encoder0(features)
        level1 = self.encoder1(F.avg_pool2d(level0, 2))
        level2 = self.encoder2(F.avg_pool2d(level1, 2))
        level3 = self.bottleneck_dropout(self.encoder3(F.avg_pool2d(level2, 2)))
        upsample = lambda value, reference: F.interpolate(value, size=reference.shape[-2:], mode='bilinear', align_corners=False)
        decoded2 = self.decoder2(torch.cat([upsample(level3, level2), level2], dim=1))
        decoded1 = self.decoder1(torch.cat([upsample(decoded2, level1), level1], dim=1))
        decoded0 = self.decoder0(torch.cat([upsample(decoded1, level0), level0], dim=1))
        residual = self.residual_bound * torch.tanh(self.output(decoded0))
        if self.gate is not None:
            residual = residual * torch.sigmoid(self.gate(decoded0))
        return residual
