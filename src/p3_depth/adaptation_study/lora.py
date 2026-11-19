"""Adapter-only UniDepth adaptation using the unchanged official infer body."""
import inspect
import math
import re
import torch
from torch import nn
from .lora_numerics import install


class LoRALinear(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.rank = rank
        self.lora_A = nn.Linear(base.in_features, rank, bias=False,
                                device=base.weight.device, dtype=base.weight.dtype)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False,
                                device=base.weight.device, dtype=base.weight.dtype)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, inputs):
        # alpha=rank, so alpha/rank=1; no dropout or merged-weight approximation.
        return self.base(inputs) + self.lora_B(self.lora_A(inputs))


def inject(model, target, rank):
    assert target in ('decoder_attention','encoder_last4_attention')
    model.eval()
    encoder = model.pixel_encoder
    if target == 'encoder_last4_attention':
        assert not encoder.chunked_blocks, 'Declare target mapping for chunked encoder first'
        encoder.frozen_stages = len(encoder.blocks) - 4
    model.requires_grad_(False)
    attention_classes = {'AttentionBlock','AttentionDecoderBlock','SimpleAttention',
                         'Attention','MemEffAttention'}
    names = []
    for name, module in list(model.named_modules()):
        if module.__class__.__name__ not in attention_classes:
            continue
        if target == 'decoder_attention':
            eligible = name.startswith('pixel_decoder.')
        else:
            match = re.match(r'pixel_encoder\.blocks\.(\d+)\.',name)
            eligible = bool(match and int(match.group(1)) >= len(encoder.blocks)-4)
        if not eligible:
            continue
        for child_name, child in list(module.named_children()):
            if isinstance(child,nn.Linear) and child_name in {
                'q','kv','out','qkv','proj','q_ca','kv_ca','q_sa','kv_sa','out_ca','out_sa'}:
                setattr(module,child_name,LoRALinear(child,rank))
                names.append(name+'.'+child_name)
    assert names, 'No declared attention targets found'
    install(model)
    assert_adapters_only(model)
    return sorted(names)


def assert_adapters_only(model):
    expected = {name for name,_ in model.named_parameters()
                if '.lora_A.' in name or '.lora_B.' in name}
    actual = {name for name,p in model.named_parameters() if p.requires_grad}
    assert expected and actual == expected, (actual-expected,expected-actual)
    return sorted(expected)


def differentiable_infer(model, rgb):
    # Explicitly unwrap the official no_grad/autocast decorators. Keep its exact
    # normalization, resize, camera prediction and point-to-Z postprocessing.
    # The model remains in evaluation mode; only adapters require gradients.
    body = inspect.unwrap(type(model).infer)
    assert body is not type(model).infer, 'Expected the pinned official decorators'
    with torch.autocast('cuda',dtype=torch.float16):
        return body(model,rgb)['depth'].float()


def adapter_state(model):
    names = assert_adapters_only(model)
    state = dict(model.named_parameters())
    return {name:state[name].detach().cpu().clone() for name in names}


def load_adapters(model,state):
    assert set(state) == set(assert_adapters_only(model))
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name,value in state.items():
            parameters[name].copy_(value)
