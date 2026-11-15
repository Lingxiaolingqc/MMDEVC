import torch
from torch import nn

from depth_project import metric_adaptation as adaptation


def test_lora_qv_preserves_initial_qkv_and_frozen_base():
    torch.manual_seed(7)
    base = nn.Linear(12, 36)
    values = torch.randn(2, 8, 12)
    expected = base(values).detach()
    layer = adaptation.LowRankQKV(base, rank=4)
    assert torch.equal(layer(values), expected)
    assert all(not parameter.requires_grad for parameter in layer.base.parameters())
    optimizer = torch.optim.SGD([parameter for parameter in layer.parameters() if parameter.requires_grad], lr=0.1)
    layer(values).square().mean().backward()
    optimizer.step()
    actual = layer(values)
    assert torch.equal(actual[..., 12:24], expected[..., 12:24])
    assert not torch.equal(actual[..., :12], expected[..., :12])
    assert not torch.equal(actual[..., 24:], expected[..., 24:])
    assert torch.equal(layer.base(values), expected)








def test_cached_projection_head_preserves_radius_geometry_and_gradients():
    head = adaptation.ProjectionDepth(input_channels=8)
    values = torch.randn(1, 8, 12, 16)
    rays = torch.full((1, 1, 24, 32), 0.8)
    output = head(values, rays, output_hw=(24, 32))
    assert output.shape == (1, 1, 24, 32)
    assert (output > 0).all()
    output.mean().backward()
    assert head.to_depth_lr.weight.grad.abs().sum() > 0
    assert head.to_depth_hr[-1].weight.grad.abs().sum() > 0
