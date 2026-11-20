"""Adapted from kwanyun/StyleSketch@0214b46d51a7cd6e50674b4e4461adb3f720aedc.

Preserves conv_block, up_conv, Attention_block and CBAM operations. UniDepth
feature pairing, reduced widths and the shared additive depth head are adaptations.
See docs/stylesketch_residual_plan.md for exact attribution and differences.
"""
import torch
from torch import nn
from torch.nn import functional as F
from p3_depth.residual.model import Refiner


class ConvBlock(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, padding=1, bias=False), nn.BatchNorm2d(outputs),
            nn.LeakyReLU(.2, inplace=True),
            nn.Conv2d(outputs, outputs, 3, padding=1, bias=False), nn.BatchNorm2d(outputs),
            nn.LeakyReLU(.2, inplace=True))

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2),
            nn.Conv2d(inputs, outputs, 3, padding=1, bias=False), nn.BatchNorm2d(outputs),
            nn.LeakyReLU(.2, inplace=True))

    def forward(self, x):
        return self.up(x)


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=False), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=False), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.LeakyReLU(.2, inplace=True)

    def forward(self, g, x):
        return x * self.psi(self.relu(self.W_g(g) + self.W_x(x)))


class ChannelGate(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.mlp = nn.Sequential(nn.Flatten(), nn.Linear(channels, channels // reduction_ratio),
                                 nn.ReLU(), nn.Linear(channels // reduction_ratio, channels))

    def forward(self, x):
        size = x.shape[-2:]
        weight = self.mlp(F.avg_pool2d(x, size, stride=size)) + self.mlp(F.max_pool2d(x, size, stride=size))
        return x * weight.sigmoid()[:, :, None, None]


class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(1, eps=1e-5, momentum=.01, affine=True)

    def forward(self, x):
        compressed = torch.cat([x.max(dim=1, keepdim=True)[0], x.mean(dim=1, keepdim=True)], dim=1)
        return x * self.bn(self.conv(compressed)).sigmoid()


class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ChannelGate = ChannelGate(channels)
        self.SpatialGate = SpatialGate()

    def forward(self, x):
        return self.SpatialGate(self.ChannelGate(x))


class StyleSketchResidual(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        widths = config['widths']
        self.start = nn.Conv2d(config['channels']['init'], config['start_width'], 1)
        self.projections = nn.ModuleDict({name: nn.Conv2d(config['channels'][name], width // 2, 1)
            for pair, width in zip(config['pairs'], widths) for name in pair})
        self.ups = nn.ModuleList([UpConv(a, b) for a, b in zip([config['start_width'], *widths[:-1]], widths)])
        self.attentions = nn.ModuleList([AttentionBlock(w, w, w // 2) for w in widths])
        # Official SketchGenerator fuses intermediate levels twice, then applies
        # CBAM directly to the final concatenation, without a final conv_block.
        self.fusions = nn.ModuleList([ConvBlock(2*w, w) for w in widths[:-1]])
        self.cbam = CBAM(2 * widths[-1])
        self.output_projection = nn.Conv2d(2 * widths[-1], 16, 1)
        self.body = Refiner(4 + 16, 'cnn', 0)

    def forward(self, inputs, internal):
        hidden = self.start(internal['init'])
        for i, (pair, up, attention) in enumerate(zip(self.config['pairs'], self.ups, self.attentions)):
            left, right = [self.projections[name](internal[name]) for name in pair]
            assert left.shape[-2:] == right.shape[-2:], 'Only native same-scale features may be paired'
            current = torch.cat([left, right], dim=1)
            hidden = up(hidden)
            assert hidden.shape == current.shape, (hidden.shape, current.shape)
            hidden = torch.cat([attention(hidden, current), hidden], dim=1)
            if i < len(self.fusions):
                hidden = self.fusions[i](hidden)
        hidden = self.output_projection(self.cbam(hidden))
        hidden = F.interpolate(hidden, inputs.shape[-2:], mode='bilinear', align_corners=False)
        return self.body(torch.cat([inputs, hidden], dim=1))
