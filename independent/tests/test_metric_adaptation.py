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


def test_radius_to_z_uses_camera_ray_direction_and_output_geometry():
    radius = torch.full((1, 1, 8, 10), 5.0, requires_grad=True)
    rays = torch.zeros((1, 3, 8, 10))
    rays[:, 0] = 0.6
    rays[:, 2] = 0.8
    depth = adaptation.radius_to_z(radius, rays, output_hw=(6, 8), padding=(0, 0, 0, 0))
    assert depth.shape == (1, 1, 6, 8)
    torch.testing.assert_close(depth, torch.full_like(depth, 4.0))
    depth.mean().backward()
    assert radius.grad.abs().sum() > 0


def test_feature_alignment_preserves_constant_and_handles_padding():
    feature = torch.ones(1, 4, 5, 7)
    aligned = adaptation.align_depth_features(feature, padded_hw=(70, 98), padding=(7, 7, 5, 5), output_hw=(24, 32))
    assert aligned.shape == (1, 4, 24, 32)
    torch.testing.assert_close(aligned, torch.ones_like(aligned))


def test_projection_selection_freezes_every_other_parameter():
    model = nn.Module()
    model.pixel_encoder = nn.Linear(4, 4)
    model.pixel_decoder = nn.Module()
    model.pixel_decoder.depth_layer = nn.Module()
    model.pixel_decoder.depth_layer.to_depth_lr = nn.Conv2d(4, 2, 1)
    model.pixel_decoder.depth_layer.to_depth_hr = nn.Conv2d(2, 1, 1)
    names = adaptation.configure_projections(model)
    assert names
    assert all("to_depth_" in name for name, parameter in model.named_parameters() if parameter.requires_grad)
    assert all(not parameter.requires_grad for parameter in model.pixel_encoder.parameters())


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
