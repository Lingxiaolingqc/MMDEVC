"""Discarded fitting-only engineering checks, then freeze the entire fixed queue."""
import json
import torch
from .common import freeze, check_freeze
from .data import load_pack
from .model import make_model
from p3_depth.residual_inputs.evaluation import batch, evaluate
from p3_depth.residual.model import corrected_depth, loss_function
from p3_depth.residual.records import write_json, now


def buffers(model):
    return {name:value.detach().clone() for name,value in model.named_buffers()}


def same_buffers(before,model):
    return all(torch.equal(before[name],value) for name,value in model.named_buffers())


def run(root,study):
    out=root/study['output']
    if (out/'design_frozen.json').exists():
        check_freeze(root,study)
        return json.loads((out/'preflight.json').read_text())
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    records={}
    for name in study['queue']:
        pack,features,internal=load_pack(root,study,name); cfg=study['candidates'][name]
        torch.manual_seed(7320112)
        model=make_model(4,cfg,internal).cuda()
        inputs,latent,bases,truths,masks,_=batch(pack,features,internal,pack['original'],pack['fitting'][:8])
        assert all(not value.requires_grad for value in latent.values())
        model.eval()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16): zero=model(inputs,latent)
        assert zero.shape==(8,1,240,320) and torch.count_nonzero(zero)==0
        assert torch.equal(corrected_depth(bases,zero,'additive'),bases.clamp_min(.001))
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.001)
        reached={key:False for key in dict(model.named_children())}; steps=[]
        torch.cuda.reset_peak_memory_stats()
        for step in range(3):
            model.train(); optimizer.zero_grad(set_to_none=True)
            # FP32 gradient-connectivity check; formal training remains FP16 AMP.
            pred=corrected_depth(bases,model(inputs,latent),'additive')
            loss=loss_function(pred,truths,masks); loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            assert torch.isfinite(norm) and torch.isfinite(loss)
            for key,value in model.named_parameters():
                assert value.grad is not None and torch.isfinite(value.grad).all(),(name,key)
                reached[key.split('.')[0]] |= bool(torch.count_nonzero(value.grad))
            optimizer.step()
            steps.append({'step':step+1,'fitting_loss':float(loss.detach()),'gradient_norm':float(norm)})
        # Identity gate lists have no parameters, so only require parameterized groups.
        required={key.split('.')[0] for key,_ in model.named_parameters()}
        assert all(reached[key] for key in required),(name,reached)
        before=buffers(model)
        evaluate(model,cfg,pack,features,internal,pack['original'],pack['validation'][:8],8)
        assert same_buffers(before,model)
        records[name]={'status':'pass','at':now(),'parameters':sum(p.numel() for p in model.parameters()),
            'initial_residual_exact_zero':True,'initial_depth_equals_original':True,
            'gradient_groups_reached':reached,'gradient_check_precision':'fp32; training uses fp16 AMP',
            'steps':steps,'validation_buffers_unchanged':True,'feature_tensors_detached':True,
            'foundation_frozen_and_hash_unchanged':pack['feature_metadata']['foundation_frozen'],
            'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'engineering_weights_discarded':True}
        print('PREFLIGHT '+name+' parameters='+str(records[name]['parameters']),flush=True)
        del model,optimizer,inputs,latent,bases,truths,masks,zero,pred,loss
        torch.cuda.empty_cache()
    write_json(out/'preflight.json',records)
    freeze(root,study,records)
    return records
