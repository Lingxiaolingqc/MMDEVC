"""Reuse prior residual bodies; explicitly adapt a constant ViT grid when needed."""
import torch
from torch.nn import functional as F
from p3_depth.residual_inputs.model import make_model as make_flat
from p3_depth.adaptation_study.model import ProgressiveRefiner
from p3_depth.stylesketch_residual.model import StyleSketchResidual


class EncoderProgressive(ProgressiveRefiner):
    def __init__(self, inputs, config, channels):
        super().__init__(inputs, config, channels)
        self.scales = config['grid_scales']

    def forward(self, inputs, internal):
        projected = []
        for name, scale in zip(self.branches, self.scales):
            value = self.projections[name](internal[name])
            if scale != 1:
                value = F.interpolate(value, scale_factor=scale, mode='bilinear', align_corners=False)
            projected.append(value)
        hidden = self.start(projected[0])
        for value, fusion in zip(projected[1:], self.fusions):
            hidden = F.interpolate(hidden, value.shape[-2:], mode='bilinear', align_corners=False)
            hidden = fusion(torch.cat([hidden, value], dim=1))
        hidden = F.interpolate(hidden, inputs.shape[-2:], mode='bilinear', align_corners=False)
        return self.body(torch.cat([inputs, hidden], dim=1))


class EncoderStyleSketch(StyleSketchResidual):
    """Same learned SS1 modules; projected ViT pairs are resized to each up-conv grid.

    These are constructed resolutions, not native encoder multiscale features.
    Each branch comes from a different real block; no tensor is duplicated.
    """
    def forward(self, inputs, internal):
        hidden = self.start(internal['init'])
        for i, (pair, up, attention) in enumerate(zip(self.config['pairs'], self.ups, self.attentions)):
            left, right = [self.projections[name](internal[name]) for name in pair]
            assert left.shape[-2:] == right.shape[-2:]
            hidden = up(hidden)
            current = F.interpolate(torch.cat([left, right], 1), hidden.shape[-2:], mode='bilinear', align_corners=False)
            hidden = torch.cat([attention(hidden, current), hidden], dim=1)
            if i < len(self.fusions):
                hidden = self.fusions[i](hidden)
        hidden = self.output_projection(self.cbam(hidden))
        hidden = F.interpolate(hidden, inputs.shape[-2:], mode='bilinear', align_corners=False)
        return self.body(torch.cat([inputs, hidden], dim=1))


def make_model(inputs, config, internal):
    if config['fusion'] == 'stylesketch':
        return EncoderStyleSketch(config)
    if config['fusion'] == 'flat':
        return make_flat(inputs, config, internal)
    return EncoderProgressive(inputs, config, {k: v.shape[1] for k, v in internal.items()})
