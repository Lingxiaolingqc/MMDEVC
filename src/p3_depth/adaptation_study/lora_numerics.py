"""Training-only finite ray-angle gradients; official inference is unchanged."""
from types import MethodType
import torch

PROTOCOL = 'finite_rays_v2'


class FiniteAcos(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        ctx.save_for_backward(value)
        return torch.acos(value)

    @staticmethod
    def backward(ctx, gradient):
        value, = ctx.saved_tensors
        work = value.double() if value.dtype == torch.float64 else value.float()
        # acos has an infinite derivative at +/-1. Normalized camera rays can
        # round to these endpoints. Bound that derivative without changing the
        # forward values. This is explicitly a surrogate gradient at the poles.
        denominator = (1 - work.square()).clamp_min(torch.finfo(work.dtype).eps).sqrt()
        return (-gradient.to(work.dtype) / denominator).to(value.dtype)


def _embed_rays(self, rays):
    if not torch.is_grad_enabled():
        return self._official_embed_rays(rays)
    from unidepth.utils.geometric import flat_interpolate
    from unidepth.utils.positional_embedding import generate_fourier_features
    # Adapted from UniDepthV2 DepthHead.embed_rays (Luigi Piccinelli,
    # CC-BY-NC-4.0); only the acos backward rule differs from the pinned source.
    value = flat_interpolate(rays, old=self.original_shapes, new=self.shapes, antialias=True)
    value = value / torch.norm(value, dim=-1, keepdim=True).clip(min=1e-4)
    x, y, z = value[..., 0], value[..., 1], value[..., 2]
    polar = FiniteAcos.apply(z)
    x_clipped = x.abs().clip(min=1e-3) * (2 * (x >= 0).int() - 1)
    azimuth = torch.atan2(y, x_clipped)
    return generate_fourier_features(torch.stack([polar, azimuth], dim=-1),
                                    dim=self.hidden_dim, max_freq=max(self.shapes)//2,
                                    use_log=True, cat_orig=False)


def install(model):
    head = model.pixel_decoder.depth_layer
    if not hasattr(head, '_official_embed_rays'):
        head._official_embed_rays = head.embed_rays
        head.embed_rays = MethodType(_embed_rays, head)
