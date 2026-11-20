"""Engineering-only module parity, real-feature gradients and BN isolation."""
import ast
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from .model import ConvBlock, UpConv, AttentionBlock, CBAM, StyleSketchResidual
from .data import load_pack
from .common import freeze, audit_references
from p3_depth.residual_inputs.evaluation import batch, evaluate
from p3_depth.residual.model import corrected_depth, loss_function
from p3_depth.residual.records import write_json, now


def buffers(model):
    return {name: value.detach().clone() for name, value in model.named_buffers()}


def same_buffers(before, model):
    return all(torch.equal(before[name], value) for name, value in model.named_buffers())


def official_parity(path):
    tree = ast.parse(path.read_text(encoding='utf8'))
    selected = ast.Module(body=[node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))], type_ignores=[])
    namespace = {'torch': torch, 'nn': nn, 'F': F, 'math': math}
    exec(compile(selected, str(path), 'exec'), namespace)
    torch.manual_seed(173)
    comparisons = []
    for official, adapted, args in [('conv_block', ConvBlock, (8, 16)), ('up_conv', UpConv, (8, 16)),
                                    ('Attention_block', AttentionBlock, (16, 16, 8)), ('CBAM', CBAM, (32,))]:
        a, b = namespace[official](*args).double(), adapted(*args).double()
        def mapped(name):
            return name.replace('SpatialGate.conv.', 'SpatialGate.spatial.conv.').replace('SpatialGate.bn.', 'SpatialGate.spatial.bn.')
        b.load_state_dict({name: a.state_dict()[mapped(name)] for name in b.state_dict()})
        for mode in (True, False):
            a.train(mode); b.train(mode)
            count, channels = (2, 16) if official == 'Attention_block' else (1, args[0])
            x = [torch.randn(2, channels, 8, 10, dtype=torch.float64, requires_grad=True) for _ in range(count)]
            y = [v.detach().clone().requires_grad_(True) for v in x]
            a.zero_grad(); b.zero_grad()
            left, right = a(*x), b(*y)
            assert torch.allclose(left, right, atol=1e-12, rtol=1e-10), official
            weight = torch.randn_like(left)
            (left*weight).sum().backward(); (right*weight).sum().backward()
            max_grad = max(float((u.grad-v.grad).abs().max()) for u, v in zip(x, y))
            for name, value in b.named_parameters():
                original = dict(a.named_parameters())[mapped(name)]
                assert torch.allclose(original.grad, value.grad, atol=1e-10, rtol=1e-9), (official, name)
            assert max_grad < 1e-10
            comparisons.append({'module': official, 'training': mode, 'max_output_difference': float((left-right).detach().abs().max()),
                                'max_input_gradient_difference': max_grad, 'parameter_gradient_parity': True})
    return comparisons


def run(root, study):
    out = root/study['output']
    assert not (out/'design_frozen.json').exists(), 'Preflight is already frozen'
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    parity = official_parity(out/'official/conv.py')
    audit_references(root, study)
    pack, features, internal = load_pack(root, study)
    old_features = json.loads((root/'results/fusion_lora_records_20261119/retained_cache_metadata/stage_cache__development__metadata.json').read_text())
    shared_hashes = {name: pack['feature_metadata']['sha256'][name+'.npy'] for name in ('init', 'mid', 'late')}
    assert all(digest == old_features['sha256'][name+'.npy'] for name, digest in shared_hashes.items())
    torch.manual_seed(7320112)
    model = StyleSketchResidual(study['candidate']).cuda()
    inputs, latent, bases, truths, masks, bands = batch(pack, features, internal, pack['original'], pack['fitting'][:8])
    model.eval()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        zero = model(inputs, latent)
    assert torch.count_nonzero(zero) == 0
    assert torch.equal(corrected_depth(bases, zero, 'additive'), bases.clamp_min(.001))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.001)
    scaler = torch.amp.GradScaler('cuda')
    torch.cuda.reset_peak_memory_stats()
    steps = []
    gradient_groups = ('start', 'projections', 'ups', 'attentions', 'fusions', 'cbam', 'output_projection', 'body')
    reached = {key: False for key in gradient_groups}
    for step in range(3):
        model.train(); optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            residual = model(inputs, latent)
        prediction = corrected_depth(bases, residual, 'additive')
        loss = loss_function(prediction, truths, masks)
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        assert torch.isfinite(grad) and torch.isfinite(loss)
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            group = name.split('.')[0]
            reached[group] |= bool(torch.count_nonzero(parameter.grad))
        scaler.step(optimizer); scaler.update()
        steps.append({'step': step+1, 'fitting_loss': float(loss.detach()), 'gradient_norm': float(grad), 'amp_scale': scaler.get_scale()})
    assert all(reached.values()), reached
    assert all(not value.requires_grad for value in latent.values())
    before = buffers(model)
    evaluate(model, study['candidate'], pack, features, internal, pack['original'], pack['validation'][:8], 8)
    assert same_buffers(before, model), 'Validation changed BN running statistics'
    # Evaluation is independent of the other images in a batch (FP16 convolution
    # algorithms can differ slightly). Buffer immutability is an exact check.
    model.eval()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        a = model(inputs, latent)[:1]
        b = model(inputs[:1], {k: v[:1] for k, v in latent.items()})
    composition_difference = float((a-b).abs().max())
    assert composition_difference < .003
    result = {'status': 'pass', 'at': now(), 'official_module_parity': parity,
              'parameters': sum(p.numel() for p in model.parameters()),
              'initial_residual_exact_zero': True, 'initial_depth_equals_original': True,
              'gradient_groups_reached': reached, 'steps': steps,
              'validation_bn_buffers_unchanged': True, 'eval_batch_composition_max_difference': composition_difference,
              'feature_tensors_detached': True, 'foundation_state_unchanged': pack['feature_metadata']['foundation_frozen'],
              'F3_P3_G3_shared_feature_files_byte_identical': shared_hashes,
              'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
              'engineering_weights_discarded': True, 'validation_not_used_for_architecture_selection': True}
    write_json(out/'preflight.json', result)
    freeze(root, study, result)
    print(json.dumps(result), flush=True)
