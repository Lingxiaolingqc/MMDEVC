"""Compare flat, progressive, gated and parameter-matched common-grid fusion."""
import torch
from torch import nn
from torch.nn import functional as F
from p3_depth.residual.model import Refiner
from p3_depth.residual_inputs.model import make_model as make_flat


class ProgressiveRefiner(nn.Module):
    def __init__(self, inputs, config, channels):
        super().__init__()
        self.branches = config['branches']
        self.zero = config.get('zero_branches', [])
        self.common_grid = config['fusion'] == 'common_grid'
        self.gated = config['fusion'] == 'gated'
        widths = config['widths']
        assert sum(widths) == 16
        self.projections = nn.ModuleDict({
            name: nn.Conv2d(channels[name], width, 1)
            for name, width in zip(self.branches, widths)
        })
        self.start = nn.Sequential(nn.Conv2d(widths[0], 16, 3, padding=1), nn.GELU())
        self.fusions = nn.ModuleList([
            nn.Sequential(nn.Conv2d(16 + width, 16, 3, padding=1), nn.GELU())
            for width in widths[1:]
        ])
        self.gates = nn.ModuleList([
            nn.Conv2d(16, 1, 1) if self.gated else nn.Identity()
            for _ in widths[1:]
        ])
        self.body = Refiner(inputs + 16, 'cnn', 0)

    def forward(self, inputs, internal):
        projected = []
        for name in self.branches:
            value = internal[name]
            if name in self.zero:
                value = torch.zeros_like(value)
            value = self.projections[name](value)
            if self.common_grid:
                value = F.interpolate(value, inputs.shape[-2:], mode='bilinear', align_corners=False)
            projected.append(value)
        hidden = self.start(projected[0])
        for value, fusion, gate in zip(projected[1:], self.fusions, self.gates):
            hidden = F.interpolate(hidden, value.shape[-2:], mode='bilinear', align_corners=False)
            if self.gated:
                value = value * torch.sigmoid(gate(hidden))
            hidden = fusion(torch.cat([hidden, value], dim=1))
        hidden = F.interpolate(hidden, inputs.shape[-2:], mode='bilinear', align_corners=False)
        return self.body(torch.cat([inputs, hidden], dim=1))


def make_model(inputs, config, internal):
    if config['fusion'] == 'flat':
        return make_flat(inputs, config, internal)
    channels = {name: array.shape[1] for name, array in internal.items()}
    return ProgressiveRefiner(inputs, config, channels)
