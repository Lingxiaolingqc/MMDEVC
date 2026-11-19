"""Diverse fitting-batch numerical preflight; no validation-based design selection."""
import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path
import torch
import yaml
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from p3_depth.adaptation_study.common import load, snapshot
from p3_depth.adaptation_study.lora import inject, differentiable_infer, assert_adapters_only
from p3_depth.adaptation_study.lora_numerics import PROTOCOL
from p3_depth.adaptation_study.lora_training import prepare_pack, training_batch, frozen_digest
from p3_depth.models.runtime import ImageModel
from p3_depth.residual.model import loss_function
from p3_depth.residual.job_schedule import segments
from p3_depth.residual.records import write_json, sha, now, append_json
from p3_depth.residual.locking import study_lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--steps', type=int, default=200)
    args = parser.parse_args()
    study, _ = load(ROOT)
    out = ROOT/study['output']/'engineering/lora'/args.tag
    assert not (out/'result.json').exists(), 'Preserve previous engineering records'
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    split = json.loads((ROOT/study['split']).read_text())
    settings = yaml.safe_load((ROOT/'configs/paths.local.yaml').read_text())
    rows = []
    with study_lock(ROOT/study['output']):
        snapshot(ROOT, out)
        pack = prepare_pack(ROOT, study)
        positions = {value:i for i,value in enumerate(pack['ids'])}
        fits = [positions[i] for i in split['fit_ids']]
        schedule = {'batch_size':8, 'max_steps':args.steps, 'evaluate_every_steps':args.steps}
        batches = next(segments(fits, 320112, schedule))[1]
        for target in study['lora']['targets']:
            for rank in study['lora']['ranks']:
                adapter = ImageModel('unidepth', ROOT, settings, 'cuda', precision='fp16', threads=4)
                model = adapter.model
                rgb, _, _ = training_batch(pack, batches[0])
                with torch.no_grad():
                    official = model.infer(rgb)['depth'].float()
                torch.manual_seed(320112)
                names = inject(model, target, rank)
                original = frozen_digest(model)
                with torch.no_grad():
                    zero = differentiable_infer(model, rgb)
                difference = float((official-zero).abs().max())
                assert difference == 0, ('Zero-adapter parity', target, rank, difference)
                parameters = [p for p in model.parameters() if p.requires_grad]
                optimizer = torch.optim.AdamW(parameters, lr=3e-4, weight_decay=.001)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2000, eta_min=1e-5)
                scaler = torch.amp.GradScaler('cuda', init_scale=128)
                observed = set()
                records = []
                torch.cuda.reset_peak_memory_stats()
                tick = time.perf_counter()
                for step, selected in enumerate(batches, 1):
                    optimizer.zero_grad(set_to_none=True)
                    rgb, truth, mask = training_batch(pack, selected)
                    depth = differentiable_infer(model, rgb)
                    loss = loss_function(depth, truth, mask, 'abs_rel', torch.zeros_like(mask))
                    assert torch.isfinite(loss) and torch.isfinite(depth).all()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    grads = {n:p.grad for n,p in model.named_parameters() if p.requires_grad}
                    assert all(g is not None for g in grads.values())
                    bad = [n for n,g in grads.items() if not bool(torch.isfinite(g).all())]
                    if not bad:
                        observed.update(n for n,g in grads.items() if bool((g != 0).any()))
                        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                    before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    assert scaler.get_scale() >= 1, 'AMP scale collapse'
                    records.append({'step':step, 'ids':[pack['ids'][i] for i in selected],
                                    'loss':float(loss.detach()), 'scale_before':before,
                                    'scale_after':scaler.get_scale(), 'nonfinite_parameters':bad})
                    if step % 50 == 0:
                        print(json.dumps({'target':target,'rank':rank,'step':step,
                                          'scale':scaler.get_scale(),'skips':sum(bool(x['nonfinite_parameters']) for x in records)}), flush=True)
                assert observed == set(grads), ('Always-zero adapter gradients', set(grads)-observed)
                assert frozen_digest(model) == original
                assert_adapters_only(model)
                row = {'target':target, 'rank':rank, 'learning_rate':3e-4, 'records':records,
                       'zero_adapter_max_abs_difference':difference,
                       'frozen_parameters_unchanged':True, 'all_adapters_received_nonzero_gradients':True,
                       'target_paths':names,'seconds':time.perf_counter()-tick,
                       'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20}
                rows.append(row)
                write_json(out/(target+'_r'+str(rank)+'.json'), row)
                del parameters, optimizer, scheduler, scaler, model, adapter, official, zero, depth, loss, grads
                gc.collect()
                torch.cuda.empty_cache()
        write_json(out/'result.json', {'status':'pass','at':now(),'numerical_protocol':PROTOCOL,
                   'split_sha256':sha(ROOT/study['split']),'rows':rows,
                   'scope':'Engineering preflight on fitting data only; weights discarded; not method selection.'})
        print('PASS diverse LoRA preflight', flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        study, _ = load(ROOT)
        append_json(ROOT/study['output']/'engineering_failures.jsonl',
                    {'command':sys.argv,'error':traceback.format_exc(),'at':now()})
        raise
