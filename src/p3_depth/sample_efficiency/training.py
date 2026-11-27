"""Recorded training, early stopping, independent best/final scoring and exact resume."""
import json
import time
import random
import traceback
import numpy as np
import torch
from .protocol import load,outdir,name,check_freeze,batch_order,update_stop,convergence
from .data import Pack
from .model import Pipeline
from p3_depth.residual.records import sha,identity,write_json,append_json,now
from p3_depth.residual.training import atomic_save
from p3_depth.residual.model import loss_function
from p3_depth.residual.evaluation import score_image,save_evaluation
from p3_depth.adaptation_study.lora_training import frozen_digest

def setup(seed):
    torch.set_num_threads(4);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def evaluate(pipe,pack,ids,directory=None,fixed_ids=()):
    before=pipe.buffers();rows=[];fixed={};tick=time.perf_counter()
    if pipe.refiner:pipe.refiner.eval()
    with torch.no_grad():
        for i in pack.positions(ids):
            prediction=pipe.predict(pack,[i])[0,0]
            assert torch.isfinite(prediction).all(),'Nonfinite prediction'
            if directory is not None:
                pred=prediction.cpu().numpy();rows.append(score_image(pred,pack,i))
                if pack['ids'][i]+1 in fixed_ids:fixed[str(pack['ids'][i]+1)]=pred
            else:
                p=prediction.clamp(.001,10).double()
                t=pack.tensor('truth',[i])[0].double();mask=pack.tensor('mask',[i])[0]
                safe=torch.where(mask,t,1.);n=mask.sum();error=p-safe
                rows.append({'abs_rel':float(((error.abs()/safe)*mask).sum()/n),
                             'rmse_m':float(((error.square()*mask).sum()/n).sqrt()),
                             'delta1':float(((torch.maximum(p/safe,safe/p)<1.25)*mask).sum()/n)})
    after=pipe.buffers()
    assert before.keys()==after.keys() and all(torch.equal(before[n],after[n]) for n in before),'Evaluation updated model buffers'
    if directory is not None:
        metrics=save_evaluation(directory,rows)
        if fixed:np.savez_compressed(directory/'fixed_predictions.npz',**fixed)
    else:metrics={k:float(np.mean([r[k] for r in rows])) for k in ('abs_rel','rmse_m','delta1')}
    return metrics,time.perf_counter()-tick

def snapshot(pipe,optimizer,scheduler,scaler,history,stop,step,exposures,skips,job_hash,elapsed):
    return {'pipeline':pipe.state(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
            'scaler':scaler.state_dict(),'history':history,'stop':stop,'step':step,'exposures':exposures,
            'skips':skips,'job_hash':job_hash,'elapsed':elapsed,'rng_cpu':torch.get_rng_state(),
            'rng_cuda':torch.cuda.get_rng_state_all(),'rng_numpy':np.random.get_state(),'rng_python':random.getstate()}

def restore(saved,pipe,optimizer,scheduler,scaler):
    pipe.restore(saved['pipeline']);optimizer.load_state_dict(saved['optimizer'])
    scheduler.load_state_dict(saved['scheduler']);scaler.load_state_dict(saved['scaler'])
    torch.set_rng_state(saved['rng_cpu']);torch.cuda.set_rng_state_all(saved['rng_cuda'])
    np.random.set_state(saved['rng_numpy']);random.setstate(saved['rng_python'])

def verify_result(root,job):
    run=outdir(root)/'runs'/name(job);result=json.loads((run/'result.json').read_text())
    assert result['job']==job and result['freeze_sha256']==sha(outdir(root)/'frozen.json')
    for n,h in result['checkpoint_sha256'].items():assert sha(run/n)==h,(run,n)
    return result

def train(root,job):
    cfg=load(root);frozen=check_freeze(root);out=outdir(root);run=out/'runs'/name(job);run.mkdir(parents=True,exist_ok=True)
    if (run/'result.json').exists():return verify_result(root,job)
    started=now();append_json(run/'attempts.jsonl',{'event':'started','at':started})
    pipe=None;pack=None
    try:
        setup(job['seed']);split=json.loads((out/'subsets.json').read_text())
        fits=split['subsets'][str(job['count'])]['ids'];vals=split['validation_ids']
        assert len(fits)==job['count'] and not set(fits)&set(vals+split['test_ids'])
        parent=None;stage=None
        if job['method']=='S':
            parent_job={**job,'method':'L'};parent=verify_result(root,parent_job)
            parent_path=out/'runs'/name(parent_job)/'best.pt'
            stage=torch.load(parent_path,map_location='cpu',weights_only=False)['pipeline']
        pack=Pack(root);pipe=Pipeline(root,cfg,job['method'],job['seed'],stage)
        config={'job':job,'training':cfg['training'],'fit_ids':fits,'validation_ids':vals,
                'freeze_sha256':sha(out/'frozen.json'),'parameters':sum(p.numel() for p in pipe.parameters),
                'parent_lora_sha256':parent['checkpoint_sha256']['best.pt'] if parent else None}
        if (run/'config.json').exists():assert json.loads((run/'config.json').read_text())==config
        else:write_json(run/'config.json',config)
        job_hash=identity(config);tr=cfg['training']
        optimizer=torch.optim.AdamW(pipe.parameters,lr=tr['learning_rate'],weight_decay=tr['weight_decay'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,tr['max_steps'],eta_min=tr['minimum_learning_rate'])
        scaler=torch.amp.GradScaler('cuda',init_scale=tr['initial_amp_scale'])
        foundation_hash=frozen_digest(pipe.foundation) if pipe.foundation else None
        completed=0;exposures=0;skips=0;history=[];stop={};previous=0.
        if (run/'resume.pt').exists():
            saved=torch.load(run/'resume.pt',map_location='cpu',weights_only=False)
            assert saved['job_hash']==job_hash;restore(saved,pipe,optimizer,scheduler,scaler)
            completed,exposures,skips,history,stop,previous=[saved[k] for k in ['step','exposures','skips','history','stop','elapsed']]
            del saved;append_json(run/'events.jsonl',{'event':'resume','completed_step':completed})
        elif (run/'history.json').exists():raise RuntimeError('History exists without a complete resume state')
        torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
        write_json(run/'status.json',{'status':'running','job':job,'at':now()})
        if not history:
            initial,seconds=evaluate(pipe,pack,vals)
            history=[{'step':0,'validation':initial,'validation_seconds':seconds,'elapsed_seconds':time.perf_counter()-tick}]
            stop=update_stop({},initial['abs_rel'],0,tr)
            atomic_save(run/'best.pt',{'pipeline':pipe.state(),'step':0,'job_hash':job_hash,'validation':initial})
            atomic_save(run/'resume.pt',snapshot(pipe,optimizer,scheduler,scaler,history,stop,0,0,0,job_hash,time.perf_counter()-tick))
            write_json(run/'history.json',history)
        total_loss=0.;seen=0;train_seconds=0.
        for step,ids in enumerate(batch_order(fits,job['seed'],tr['max_steps'],tr['batch_size']),1):
            if step<=completed:continue
            if stop['stop']:break
            positions=pack.positions(ids)
            if pipe.refiner:pipe.refiner.train()
            optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();batch_tick=time.perf_counter();batch_loss=0.
            for offset in range(0,len(positions),tr['microbatch']):
                pos=positions[offset:offset+tr['microbatch']]
                prediction=pipe.predict(pack,pos)
                truth=pack.tensor('truth',pos)[:,None];mask=pack.tensor('mask',pos)[:,None]
                loss=loss_function(prediction,truth,mask)*len(pos)/len(positions)
                assert torch.isfinite(loss),'Nonfinite loss'
                scaler.scale(loss).backward();batch_loss+=float(loss.detach())*len(positions)
                del prediction,truth,mask,loss
            scaler.unscale_(optimizer)
            finite=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in pipe.parameters)
            if finite:torch.nn.utils.clip_grad_norm_(pipe.parameters,tr['clip_gradient_norm'],error_if_nonfinite=True)
            else:
                skips+=1;append_json(run/'events.jsonl',{'event':'AMP step skipped','step':step,'scale':scaler.get_scale()})
            scaler.step(optimizer);scaler.update();scheduler.step()
            if scaler.get_scale()<1:raise FloatingPointError('AMP scale below 1')
            torch.cuda.synchronize();train_seconds+=time.perf_counter()-batch_tick
            total_loss+=batch_loss;seen+=len(ids);exposures+=len(ids)
            if step%25==0:write_json(run/'progress.json',{'step':step,'exposures':exposures,'amp_skips':skips,'at':now()})
            if step%tr['evaluate_every_steps']!=0 and step<tr['max_steps']:continue
            metrics,seconds=evaluate(pipe,pack,vals)
            old_best=stop['best'];stop=update_stop(stop,metrics['abs_rel'],step,tr)
            elapsed=previous+time.perf_counter()-tick
            row={'step':step,'epoch_equivalent':exposures/len(fits),'exposures':exposures,
                 'successful_updates':step-skips,'amp_skips':skips,'amp_scale':scaler.get_scale(),
                 'train_loss':total_loss/seen,'training_seconds':train_seconds,'validation_seconds':seconds,
                 'elapsed_seconds':elapsed,'learning_rate':optimizer.param_groups[0]['lr'],
                 'validation':metrics,'early_stop':dict(stop)}
            history.append(row)
            if metrics['abs_rel']<old_best:
                atomic_save(run/'best.pt',{'pipeline':pipe.state(),'step':step,'job_hash':job_hash,'validation':metrics})
            # Checkpoint is authoritative if interruption happens before the history mirror.
            atomic_save(run/'resume.pt',snapshot(pipe,optimizer,scheduler,scaler,history,stop,step,exposures,skips,job_hash,elapsed))
            write_json(run/'history.json',history);append_json(run/'epochs.jsonl',row)
            print(name(job)+' '+json.dumps(row),flush=True)
            completed=step;total_loss=0.;seen=0;train_seconds=0.
        atomic_save(run/'final.pt',{'pipeline':pipe.state(),'step':completed,'job_hash':job_hash,'validation':history[-1]['validation']})
        final,final_seconds=evaluate(pipe,pack,vals,run/'final_validation',cfg['fixed_visual_ids'])
        assert abs(final['abs_rel']-history[-1]['validation']['abs_rel'])<1e-8
        best=torch.load(run/'best.pt',map_location='cpu',weights_only=False);pipe.restore(best['pipeline'])
        best_metrics,best_seconds=evaluate(pipe,pack,vals,run/'best_validation',cfg['fixed_visual_ids'])
        assert abs(best_metrics['abs_rel']-best['validation']['abs_rel'])<1e-8
        assert not pipe.foundation or frozen_digest(pipe.foundation)==foundation_hash,'Frozen foundation changed'
        result={'status':'complete','job':job,'freeze_sha256':config['freeze_sha256'],'job_hash':job_hash,
                'parameters':config['parameters'],'best':best_metrics,'final':final,
                'convergence':convergence(history,cfg['diagnostics']),
                'stop_reason':'patience' if stop['stop'] else 'max_steps','attempted_updates':completed,
                'successful_updates':completed-skips,'exposures':exposures,'amp_skips':skips,
                'elapsed_seconds':previous+time.perf_counter()-tick,
                'detailed_scoring_seconds':final_seconds+best_seconds,
                'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                'parent_lora_seconds':parent['elapsed_seconds'] if parent else 0,
                'parent_lora_attempted_updates':parent['attempted_updates'] if parent else 0,
                'stage_feature_policy':'online frozen L features; time included in S steps' if parent else None,
                'checkpoint_sha256':{n:sha(run/n) for n in ['best.pt','final.pt']},'finished_at':now()}
        write_json(run/'result.json',result);write_json(run/'status.json',{'status':'complete','job':job,'at':now()})
        append_json(run/'attempts.jsonl',{'event':'complete','at':now(),'started':started})
        return result
    except BaseException as exc:
        failure={'at':now(),'status':'failed','job':job,'error':repr(exc),'traceback':traceback.format_exc()}
        write_json(run/'status.json',failure);append_json(out/'failures.jsonl',failure)
        append_json(run/'attempts.jsonl',{'event':'failed','at':now(),'started':started,'error':repr(exc)})
        raise
    finally:
        if pack:pack.close()
        if pipe:pipe.close()

def baseline(root):
    cfg=load(root);check_freeze(root);out=outdir(root)/'baseline';out.mkdir(exist_ok=True)
    if (out/'metrics.json').exists():return json.loads((out/'metrics.json').read_text())
    pack=Pack(root);split=json.loads((outdir(root)/'subsets.json').read_text())
    try:
        rows=[score_image(pack['original'][i],pack,i) for i in pack.positions(split['validation_ids'])]
        metrics=save_evaluation(out,rows);write_json(out/'identity.json',{'unique_training_images':0,
             'independent_seeds':0,'shared_by':['R','L','J','S'],'source':'verified original prediction cache',
             'freeze_sha256':sha(outdir(root)/'frozen.json')})
        return metrics
    finally:pack.close()
