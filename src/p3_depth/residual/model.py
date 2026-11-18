"""Explicit input/architecture/output factors; no foundation model is trained."""
import torch
from torch import nn
from torch.nn import functional as F


def candidate(name):
    result = dict(name=name, base='original', inputs='rgb_depth', architecture='cnn', output='additive', loss='abs_rel')
    if name in ('base_scale', 'base_flip', 'base_flip_scale'):
        result['base'] = name.removeprefix('base_')
    elif name in ('depth_only', 'flip', 'confidence', 'coordinates', 'frozen_features', 'rgb_sobel', 'rgb_canny', 'depth_sobel', 'depth_canny', 'flip_disagreement', 'joint_edges'):
        result['inputs'] = name
    elif name in ('mlp', 'unet'):
        result['architecture'] = name
    elif name in ('log_output', 'bounded_log_output'):
        result['output'] = name.removesuffix('_output')
    elif name in ('gradient_loss', 'boundary_loss'):
        result['loss'] = name.removesuffix('_loss')
    elif name == 'legacy_bundle':
        result.update(base='flip_scale', inputs='legacy', architecture='legacy_unet', output='bounded_log')
    elif name != 'R0':
        raise ValueError(f'Unknown candidate {name}')
    return result


def block(inputs, outputs):
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1), nn.GroupNorm(4, outputs), nn.GELU(),
                         nn.Conv2d(outputs, outputs, 3, padding=1), nn.GroupNorm(4, outputs), nn.GELU())


class Refiner(nn.Module):
    def __init__(self, channels, architecture='cnn', internal_channels=0):
        super().__init__()
        self.projection = nn.Conv2d(internal_channels, 16, 1) if internal_channels else None
        channels += 16 if internal_channels else 0
        self.architecture = architecture
        if architecture in ('cnn', 'mlp'):
            kernel, width = (3, 32) if architecture == 'cnn' else (1, 96)
            layers = []
            for i in range(3):
                layers.extend([nn.Conv2d(channels if i == 0 else width, width, kernel, padding=kernel//2), nn.GELU()])
            self.hidden = nn.Sequential(*layers)
            self.output = nn.Conv2d(width, 1, kernel, padding=kernel//2)
        elif architecture in ('unet', 'legacy_unet'):
            widths = (8, 12, 16, 24) if architecture == 'unet' else (16, 32, 64, 96)
            self.encoders = nn.ModuleList([block(channels if i == 0 else widths[i-1], width) for i, width in enumerate(widths)])
            self.decoders = nn.ModuleList([block(widths[i+1] + widths[i], widths[i]) for i in (2, 1, 0)])
            self.output = nn.Conv2d(widths[0], 1, 1)
        else:
            raise ValueError('Unknown architecture')
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, inputs, internal=None):
        if self.projection is not None:
            if internal is None:
                raise ValueError('Missing frozen internal features')
            projected = F.interpolate(self.projection(internal), inputs.shape[-2:], mode='bilinear', align_corners=False)
            inputs = torch.cat([inputs, projected], 1)
        if self.architecture in ('cnn', 'mlp'):
            hidden = self.hidden(inputs)
        else:
            levels = []
            hidden = inputs
            for i, encoder in enumerate(self.encoders):
                hidden = encoder(F.avg_pool2d(hidden, 2) if i else hidden)
                levels.append(hidden)
            for decoder, skip in zip(self.decoders, levels[-2::-1]):
                hidden = F.interpolate(hidden, skip.shape[-2:], mode='bilinear', align_corners=False)
                hidden = decoder(torch.cat([hidden, skip], 1))
        return self.output(hidden)


def corrected_depth(base, residual, output):
    # Bound on the native grid, then interpolate, matching the old refiner's semantics.
    residual = residual.float()
    if output == 'bounded_log':
        residual = .5 * torch.tanh(residual)
    residual = F.interpolate(residual, base.shape[-2:], mode='bilinear', align_corners=False)
    if output == 'additive':
        return (base + residual).clamp_min(.001)
    if output in ('log', 'bounded_log'):
        return base * residual.exp()
    raise ValueError('Unknown output representation')


def loss_function(prediction, truth, mask, kind='abs_rel', band=None, gradient_weight=.1, boundary_weight=2.):
    prediction = prediction.float().clamp(.001, 10)
    safe_truth = torch.where(mask, truth, 1.)
    error = (prediction - safe_truth).abs() / safe_truth
    weights = mask.float()
    if kind == 'boundary':
        if band is None:
            raise ValueError('GT boundary band required for boundary loss')
        weights = weights * (1 + (boundary_weight - 1) * band)
    loss = ((error * weights).sum((1, 2, 3)) / weights.sum((1, 2, 3))).mean()
    if kind == 'gradient':
        predicted_log, truth_log = prediction.log(), safe_truth.log()
        terms = []
        for dim in (-1, -2):
            left, right = [slice(None)] * 4, [slice(None)] * 4
            left[dim], right[dim] = slice(None, -1), slice(1, None)
            left, right = tuple(left), tuple(right)
            valid = mask[left] & mask[right]
            difference = ((predicted_log[right] - predicted_log[left]) - (truth_log[right] - truth_log[left])).abs()
            terms.append((difference * valid).sum((1, 2, 3)) / valid.sum((1, 2, 3)).clamp_min(1))
        loss = loss + gradient_weight * torch.stack(terms).mean()
    elif kind not in ('abs_rel', 'boundary'):
        raise ValueError('Unknown loss')
    return loss
