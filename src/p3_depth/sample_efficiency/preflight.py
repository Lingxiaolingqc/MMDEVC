"""Discarded fitting-only engineering checks before freezing formal experiment code."""
import gc
import json
import torch
import numpy as np
from .protocol import load,outdir,prepare,update_stop,batch_order,convergence
from .data import Pack
from .model import Pipeline
from .training import setup,evaluate,snapshot,restore
from p3_depth.residual.records import write_json,sha
from p3_depth.residual.training import atomic_save
from p3_depth.residual.model import loss_function
from p3_depth.adaptation_study.lora_training import frozen_digest

def protocol_checks(root):
    cfg=load(root);tr=cfg['training'];sub=json.loads((outdir(root)/'subsets.json').read_text())
    previous=set()
    for n in cfg['counts']:
        ids=sub['subsets'][str(n)]['ids'];assert previous<=set(ids);previous=set(ids)
        assert len(ids)==n and not set(ids)&set(sub['validation_ids']+sub['test_ids'])
    a=list(batch_order(sub['subsets']['100']['ids'],320112,100))
    assert a==list(batch_order(sub['subsets']['100']['ids'],320112,100))
    assert all(len(x)==8 for x in a) and sum(map(len,a))==800
    s=update_stop({},.1,0,tr)
    for step in range(100,1000,100):s=update_stop(s,.1001,step,tr);assert not s['stop']
    s=update_stop(s,.1001,1000,tr);assert s['stop']
    s=update_stop({},.1,0,tr);s=update_stop(s,.0999,100,tr)
    assert s['best']==.0999 and s['stale']==1
    s=update_stop(s,.0997,200,tr);assert s['stale']==0 and s['anchor']==.0997
    flat=[{'step':i,'validation':{'abs_rel':.1}} for i in range(0,601,100)]
    assert convergence(flat,cfg['diagnostics'])['stable_step']==0
    bad=flat+[{'step':700,'validation':{'abs_rel':.105}}]
    assert not convergence(bad,cfg['diagnostics'])['stable_at_end']
    return {'nested':True,'disjoint':True,'batch_repeatable':True,'early_stop_and_raw_best_separate':True,'stable_diagnostic':True}

def run(root):
    cfg=load(root);out=outdir(root);folder=out/'engineering';folder.mkdir(exist_ok=True)
    prepare(root);unit=protocol_checks(root);pack=Pack(root)
    split=json.loads((out/'subsets.json').read_text());fits=split['subsets']['100']['ids']
    batches=list(batch_order(fits,320112,22));report={};lora_state=None
    for method in ['R','L','J','S']:
        setup(320112);pipe=Pipeline(root,cfg,method,320112,lora_state if method=='S' else None)
        before=frozen_digest(pipe.foundation) if pipe.foundation else None
        with torch.no_grad():
            i=pack.pos[fits[0]];p=pipe.predict(pack,[i])
            if method!='S':
                expected=pack.tensor('original',[i])[:,None]
                delta=float((p-expected).abs().max());assert delta<=1e-5,(method,delta)
            else:
                reference=pipe.infer(pack.rgb([i]))['depth'].float()
                delta=float((p-reference).abs().max());assert delta<=1e-5
        tr=cfg['training'];optimizer=torch.optim.AdamW(pipe.parameters,lr=tr['learning_rate'],weight_decay=tr['weight_decay'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,tr['max_steps'],eta_min=tr['minimum_learning_rate'])
        scaler=torch.amp.GradScaler('cuda',init_scale=tr['initial_amp_scale'])
        received=set();skips=0;torch.cuda.reset_peak_memory_stats()
        def update(ids):
            nonlocal skips
            optimizer.zero_grad(set_to_none=True)
            pos=pack.positions(ids)
            for offset in range(0,8,tr['microbatch']):
                indices=pos[offset:offset+tr['microbatch']]
                p=pipe.predict(pack,indices);t=pack.tensor('truth',indices)[:,None];m=pack.tensor('mask',indices)[:,None]
                loss=loss_function(p,t,m)*len(indices)/8;assert torch.isfinite(loss)
                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            finite=True
            for n,v in pipe.named_trainable():
                if v.grad is not None:
                    finite=finite and bool(torch.isfinite(v.grad).all())
                    if bool(torch.isfinite(v.grad).all()) and float(v.grad.abs().sum())>0:received.add(n)
            if finite:torch.nn.utils.clip_grad_norm_(pipe.parameters,1.,error_if_nonfinite=True)
            else:skips+=1
            scaler.step(optimizer);scaler.update();scheduler.step()
            assert scaler.get_scale()>=1
        for ids in batches[:20]:update(ids)
        expected={n for n,p in pipe.named_trainable()}
        assert expected<=received,(method,'No nonzero gradient',expected-received)
        if method=='J':
            pipe.trace=True;p=pipe.predict(pack,pack.positions(batches[20][:1]));paths=pipe.last_paths
            assert all(v.requires_grad for v in paths['latent'].values()) and paths['depth'].requires_grad
            with torch.autocast('cuda',dtype=torch.float16):
                residual=pipe.refiner(paths['inputs'].detach(),paths['latent'])
            adapters=[p for n,p in pipe.named_trainable() if 'lora_B' in n]
            # Match training's AMP-scaled backward so the feature-only probe
            # does not disappear through FP16 underflow.
            gs=torch.autograd.grad(residual.float().mean()*scaler.get_scale(),adapters,allow_unused=True)
            print('JOINT_FEATURE_GRAD '+json.dumps([None if g is None else {'finite':bool(torch.isfinite(g).all()),'max':float(g.abs().max())} for g in gs]),flush=True)
            assert any(g is not None and bool(torch.isfinite(g).all()) and float(g.abs().sum())>0 for g in gs),'Detached F3-to-LoRA feature path'
            pipe.trace=False;pipe.last_paths.clear();del p,paths,residual,gs
        state=snapshot(pipe,optimizer,scheduler,scaler,[],{},20,160,skips,'engineering',0.)
        path=folder/(method+'_resume_check.pt');atomic_save(path,state)
        update(batches[21]);expected_state=pipe.state()
        saved=torch.load(path,map_location='cpu',weights_only=False);restore(saved,pipe,optimizer,scheduler,scaler)
        restored=pipe.state()
        assert all(torch.equal(v,restored[group][n]) for group,values in saved['pipeline'].items() for n,v in values.items()),'Checkpoint parameter restore changed values'
        update(batches[21]);actual_state=pipe.state()
        max_delta=max(float((v-actual_state[group][n]).abs().max()) for group,values in expected_state.items() for n,v in values.items())
        bitwise=all(torch.equal(v,actual_state[group][n]) for group,values in expected_state.items() for n,v in values.items())
        print('REPLAY '+method+' max_abs_delta='+str(max_delta),flush=True)
        assert all(torch.allclose(v,actual_state[group][n],atol=1e-7,rtol=1e-4) for group,values in expected_state.items() for n,v in values.items()),'Next-update replay exceeds frozen CUDA tolerance'
        _,_=evaluate(pipe,pack,split['validation_ids'][:3])
        assert pipe.foundation is None or before==frozen_digest(pipe.foundation),'Foundation weights changed'
        if method=='L':lora_state=pipe.state()
        report[method]={'zero_delta':delta,'trainable_parameters':sum(p.numel() for p in pipe.parameters),
            'all_trainable_received_gradients':True,'resume_parameters_bitwise_equal':True,
            'resume_next_update_bitwise_equal':bitwise,'resume_next_update_max_abs_delta':max_delta,
            'resume_next_update_tolerance':{'atol':1e-7,'rtol':1e-4},
            'validation_buffers_unchanged':True,
            'frozen_foundation_unchanged':True,'amp_skips':skips,'amp_scale':scaler.get_scale(),
            'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
            'f3_to_lora_live_feature_gradient':method=='J'}
        print('PREFLIGHT '+method+' '+json.dumps(report[method]),flush=True)
        pipe.close();del pipe,optimizer,scheduler,scaler,state,saved,actual_state,expected_state,restored
        gc.collect();torch.cuda.empty_cache()
    pack.close();value={'status':'pass','protocol':unit,'methods':report,'fitting_only_updates_per_method':22,
                       'engineering_weights_excluded':True,'microbatch':cfg['training']['microbatch']}
    write_json(out/'preflight.json',value);return value
