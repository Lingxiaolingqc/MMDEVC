"""Small residual network that adapts frozen metric predictions using RGB."""
import cv2
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
    def __init__(self):
        super().__init__()
        self.encoder0 = conv_block(9, 16)
        self.encoder1 = conv_block(16, 32)
        self.encoder2 = conv_block(32, 64)
        self.encoder3 = conv_block(64, 96)
        self.decoder2 = conv_block(96 + 64, 64)
        self.decoder1 = conv_block(64 + 32, 32)
        self.decoder0 = conv_block(32 + 16, 16)
        self.output = nn.Conv2d(16, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features):
        level0 = self.encoder0(features)
        level1 = self.encoder1(F.avg_pool2d(level0, 2))
        level2 = self.encoder2(F.avg_pool2d(level1, 2))
        level3 = self.encoder3(F.avg_pool2d(level2, 2))
        upsample = lambda value, reference: F.interpolate(value, size=reference.shape[-2:], mode='bilinear', align_corners=False)
        decoded2 = self.decoder2(torch.cat([upsample(level3, level2), level2], dim=1))
        decoded1 = self.decoder1(torch.cat([upsample(decoded2, level1), level1], dim=1))
        decoded0 = self.decoder0(torch.cat([upsample(decoded1, level0), level0], dim=1))
        return .5 * torch.tanh(self.output(decoded0))
