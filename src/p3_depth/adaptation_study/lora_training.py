"""Recorded LoRA search, matched-size runs and frozen full-development refits."""
import gc
import hashlib
import json
import time
import traceback
import numpy as np
import torch
import yaml
from p3_depth.data.nyuv2 import NYUv2Labeled
from p3_depth.models.runtime import ImageModel
from p3_depth.residual.records import sha,identity,write_json,append_json,now,remove_completed_resume
from p3_depth.residual.training import atomic_save
from p3_depth.residual.evaluation import score_image,save_evaluation
from p3_depth.residual.model import loss_function
from p3_depth.residual.job_schedule import segments
from .common import snapshot,check_residual_freeze,size_ids
from .data import load_pack
from .lora import inject,differentiable_infer,adapter_state,load_adapters,assert_adapters_only
from .lora_numerics import PROTOCOL


def candidates(study):
    return {f'L_{code}_r{rank}_lr{round(lr*10000)}e4':
            {'name':f'L_{code}_r{rank}_lr{round(lr*10000)}e4','target':target,'rank':rank,'learning_rate':lr}
            for target,code in zip(study['lora']['targets'],['D','E'])
            for rank in study['lora']['ranks'] for lr in study['lora']['learning_rates']}


def check_lora_freeze(root,study):
    path=root/study['output']/'lora_frozen.json'
    value=json.loads(path.read_text(encoding='utf8'))
    for name,digest in value['source_sha256'].items():
        assert sha(root/name)==digest,('LoRA source changed after freeze',name)
    return value,sha(path)


def save_job(root,study,name,seed,phase='search',size=None,smoke=False):
    if not smoke:check_residual_freeze(root,study)
    cfg=candidates(study)[name]
    split=json.loads((root/study['split']).read_text(encoding='utf8'))
    fits,vals=split['fit_ids'],split['validation_ids']
    train={'max_steps':study['lora']['search_steps'],'evaluate_every_steps':study['lora']['evaluate_every_steps'],
           'batch_size':study['lora']['batch_size'],'microbatch':study['lora']['microbatch'],
           'learning_rate':cfg['learning_rate'],'minimum_learning_rate':cfg['learning_rate']/30,
           'weight_decay':.001,'clip_gradient_norm':1.,'initial_amp_scale':study['lora']['initial_amp_scale']}
    extra={};selection='best_validation'
    if phase=='confirm':
        choice=json.loads((root/study['output']/'lora_shortlist.json').read_text())
        assert name in choice['methods']
    if phase in ('data_size','refit'):
        frozen,digest=check_lora_freeze(root,study)
        assert name==frozen['method']
        extra['freeze_sha256']=digest;selection='fixed_final'
    if phase=='data_size':
        fits,extra['subset_design_sha256']=size_ids(root,study,size)
        train['max_steps']=study['data_size']['steps']
    elif phase=='refit':
        fits,vals=sorted(fits+vals),[]
        train['max_steps']=frozen['updates'];train['schedule_steps']=study['lora']['search_steps']
    if smoke:
        assert phase=='search'
        fits,vals=fits[:16],vals[:4]
        train.update(max_steps=2,evaluate_every_steps=1)
    suffix=(f'_n{size}' if size is not None else '')+('_smoke' if smoke else '')+'_'+PROTOCOL
    job={'run_id':f'lora_{phase}_{name}_seed{seed}'+suffix,'phase':phase,'candidate':cfg,'seed':seed,
         'fit_ids':fits,'validation_ids':vals,'training':train,'selection_point':selection,'smoke':smoke,
         'split_sha256':sha(root/study['split']),'numerical_protocol':PROTOCOL,**extra}
    path=root/study['output']/'jobs'/(job['run_id']+'.json')
    record={'job':job,'job_sha256':identity(job)}
    if path.exists():assert json.loads(path.read_text())==record
    else:write_json(path,record)
    return path


def prepare_pack(root,study):
    pack=load_pack(root,study)
    directory=root/study['output']/'rgb_cache';directory.mkdir(exist_ok=True)
    path=directory/'rgb.npy';spec={'ids':pack['ids'],'dataset_sha256':sha(root/study['data'])}
    meta_path=directory/'metadata.json'
    if meta_path.exists():
        meta=json.loads(meta_path.read_text());assert meta['spec']==spec and sha(path)==meta['sha256']
    else:
        rgb=np.lib.format.open_memmap(path,mode='w+',dtype=np.uint8,shape=(len(pack['ids']),3,480,640))
        with NYUv2Labeled(root/study['data']) as data:
            for i,index in enumerate(pack['ids']):rgb[i]=data.file['images'][index].transpose(2,1,0).transpose(2,0,1)
        rgb.flush();del rgb
        write_json(meta_path,{'spec':spec,'sha256':sha(path)})
    pack['raw_rgb']=np.load(path,mmap_mode='r')
    return pack


def training_batch(pack,positions):
    def tensor(name):return torch.from_numpy(np.ascontiguousarray(pack[name][positions])).cuda()
    return tensor('raw_rgb'),tensor('truth')[:,None],tensor('mask')[:,None]


def evaluate(model,pack,positions,detailed=False,zero_parity=False):
    rows=[];maximum=0.
    with torch.no_grad():
        for i in positions:
            rgb,truth,mask=training_batch(pack,[i])
            prediction=differentiable_infer(model,rgb)
            assert torch.isfinite(prediction).all(),'Nonfinite validation prediction'
            if zero_parity:
                delta=float(np.max(np.abs(prediction[0,0].cpu().numpy()-pack['original'][i])))
                maximum=max(maximum,delta)
            if detailed:
                rows.append(score_image(prediction[0,0].cpu().numpy(),pack,i))
            else:
                p=prediction.clamp(.001,10).double();t=truth.double()
                safe=torch.where(mask,t,1.);count=mask.sum((1,2,3));difference=p-safe
                values=[((difference.abs()/safe)*mask).sum()/count,
                        ((difference.square()*mask).sum()/count).sqrt(),
                        ((torch.maximum(p/safe,safe/p)<1.25)*mask).sum()/count]
                rows.append({key:float(value) for key,value in zip(['abs_rel','rmse_m','delta1'],values)})
    metrics={key:float(np.mean([r[key] for r in rows])) for key in ['abs_rel','rmse_m','delta1']}
    if zero_parity:
        assert maximum<=1e-5,('Zero-adapter prediction differs from frozen cache',maximum)
        metrics['zero_adapter_max_abs_difference']=maximum
    return metrics,rows


def frozen_digest(model):
    digest=hashlib.sha256()
    for name,p in model.named_parameters():
        if not p.requires_grad:
            digest.update(name.encode());digest.update(p.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def preview(model,pack,positions,output,label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(exist_ok=True)
    chosen=[positions[k] for k in sorted({0,len(positions)//2,len(positions)-1})]
    with torch.no_grad():
        for i in chosen:
            rgb,_,_=training_batch(pack,[i])
            prediction=differentiable_infer(model,rgb)[0,0].cpu().numpy()
            truth=pack['truth'][i];mask=pack['mask'][i]
            fig,axes=plt.subplots(1,5,figsize=(15,3.5),constrained_layout=True)
            axes[0].imshow(pack['raw_rgb'][i].transpose(1,2,0))
            for ax,value in zip(axes[1:4],[truth,pack['original'][i],prediction]):
                ax.imshow(value,vmin=0,vmax=10,cmap='viridis')
            error=np.where(mask,np.abs(np.clip(prediction,.001,10)-truth)/np.maximum(truth,.001),np.nan)
            axes[4].imshow(error,vmin=0,vmax=.5,cmap='magma')
            for ax,title in zip(axes,['RGB','GT (0-10m)','UniDepth (0-10m)','LoRA (0-10m)','AbsRel error (0-0.5)']):
                ax.set_title(title);ax.axis('off')
            fig.suptitle(f'{label} | validation | nyuv2_{pack["ids"][i]+1:04d}')
            fig.savefig(output/f'nyuv2_{pack["ids"][i]+1:04d}.png',dpi=140);plt.close(fig)


def run(root,study,job_path,pack,resume=False):
    record=json.loads(job_path.read_text(encoding='utf8'));job=record['job']
    assert identity(job)==record['job_sha256'] and job['split_sha256']==sha(root/study['split'])
    split=json.loads((root/study['split']).read_text())
    assert not set(split['test_ids'])&(set(job['fit_ids'])|set(job['validation_ids']))
    assert not set(job['fit_ids'])&set(job['validation_ids'])
    output=root/study['output']/'followups'/job['run_id'];output.mkdir(parents=True,exist_ok=True)
    if (output/'result.json').exists():
        result=json.loads((output/'result.json').read_text())
        assert result['job_sha256']==record['job_sha256']
        for name,digest in result['checkpoint_sha256'].items():assert sha(output/name)==digest
        print('PASS already complete '+job['run_id'],flush=True);return result
    if (output/'config.json').exists() and not resume:raise FileExistsError('Use --resume for an incomplete run')
    torch.set_num_threads(4);torch.manual_seed(job['seed']);np.random.seed(job['seed'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    train=job['training'];cfg=job['candidate']
    try:
        settings=yaml.safe_load((root/'configs/paths.local.yaml').read_text())
        adapter=ImageModel('unidepth',root,settings,'cuda',precision='fp16',threads=4)
        # ImageModel seeds official model construction internally. Restore the
        # declared run seed before constructing the randomly initialized adapters.
        torch.manual_seed(job['seed'])
        model=adapter.model;targets=inject(model,cfg['target'],cfg['rank'])
        parameters=[p for p in model.parameters() if p.requires_grad]
        initial_adapter_hash=hashlib.sha256(b''.join(
            name.encode()+value.numpy().tobytes() for name,value in adapter_state(model).items())).hexdigest()
        original_digest=frozen_digest(model)
        config={**job,'job_sha256':record['job_sha256'],'parameters':sum(p.numel() for p in parameters),
                'target_paths':targets,'foundation':adapter.metadata,'frozen_parameters_sha256':original_digest,
                'adapter_initialization_seed':job['seed'],'initial_adapter_sha256':initial_adapter_hash,
                'evaluation_batch_size':1,'model_eval_mode_during_training':True,'gt_in_inference':False}
        config_hash=identity(config)
        if (output/'config.json').exists():
            assert identity(json.loads((output/'config.json').read_text()))==config_hash
            provenance=json.loads((output/'provenance.json').read_text())
            for name,digest in provenance['source_sha256'].items():assert sha(root/name)==digest,('Resume source changed',name)
        else:
            write_json(output/'config.json',config);snapshot(root,output)
        positions={index:i for i,index in enumerate(pack['ids'])}
        fits=[positions[i] for i in job['fit_ids']];vals=[positions[i] for i in job['validation_ids']]
        optimizer=torch.optim.AdamW(parameters,lr=train['learning_rate'],weight_decay=train['weight_decay'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,train.get('schedule_steps',train['max_steps']),eta_min=train['minimum_learning_rate'])
        scaler=torch.amp.GradScaler('cuda',init_scale=train['initial_amp_scale'])
        history=[];best=float('inf');best_step=0;completed_unit=0;updates=0;exposures=0;skipped=0
        if resume and (output/'resume.pt').exists():
            saved=torch.load(output/'resume.pt',map_location='cpu',weights_only=False)
            assert saved['config_hash']==config_hash
            load_adapters(model,saved['model']);optimizer.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler']);scaler.load_state_dict(saved['scaler'])
            history,best,best_step,completed_unit,updates,exposures,skipped=[saved[k] for k in ('history','best','best_step','unit','updates','exposures','skipped')]
            torch.set_rng_state(saved['rng_cpu']);torch.cuda.set_rng_state_all(saved['rng_cuda'])
        elif resume and (output/'history.json').exists():raise ValueError('History without resume checkpoint')
        tick=time.perf_counter();previous_seconds=history[-1]['elapsed_seconds'] if history else 0.
        torch.cuda.reset_peak_memory_stats()
        append_json(root/study['output']/'runs.jsonl',{'run':job['run_id'],'status':'running','job_sha256':record['job_sha256']})
        write_json(output/'status.json',{'status':'running','at':now()})
        if not history and vals:
            initial,_=evaluate(model,pack,vals,zero_parity=True)
            write_json(output/'initial_metrics.json',initial);print('INITIAL '+job['run_id']+' '+json.dumps(initial),flush=True)
        for unit,batches in segments(fits,job['seed'],train):
            if unit<=completed_unit:continue
            total_loss=0.;seen=0
            for selected in batches:
                optimizer.zero_grad(set_to_none=True)
                batch_loss=0.
                for start in range(0,len(selected),train['microbatch']):
                    micro=selected[start:start+train['microbatch']]
                    rgb,truth,mask=training_batch(pack,micro)
                    prediction=differentiable_infer(model,rgb)
                    loss=loss_function(prediction,truth,mask,'abs_rel',torch.zeros_like(mask))
                    if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(prediction).all()):raise ValueError('Nonfinite LoRA training output')
                    scaler.scale(loss*len(micro)/len(selected)).backward()
                    batch_loss+=float(loss.detach())*len(micro)
                scaler.unscale_(optimizer)
                assert all(p.grad is not None for p in parameters), 'Missing adapter gradient'
                finite=bool(torch.stack([torch.isfinite(p.grad).all() for p in parameters]).all())
                if finite:
                    norm=torch.nn.utils.clip_grad_norm_(parameters,train['clip_gradient_norm'],error_if_nonfinite=True)
                else:
                    skipped+=1
                    append_json(output/'events.jsonl',{'event':'AMP nonfinite gradients; optimizer step skipped','update':updates+1,'scale':scaler.get_scale()})
                scaler.step(optimizer);scaler.update();scheduler.step()
                if scaler.get_scale()<1:
                    raise FloatingPointError('AMP scale collapsed below 1; exclude run and diagnose before continuing')
                updates+=1;exposures+=len(selected);seen+=len(selected);total_loss+=batch_loss
                if updates%50==0:
                    progress={'run':job['run_id'],'updates':updates,'image_exposures':exposures,'amp_skipped':skipped,
                              'amp_scale':scaler.get_scale(),'elapsed_seconds':previous_seconds+time.perf_counter()-tick}
                    write_json(output/'progress.json',progress);print('PROGRESS '+json.dumps(progress),flush=True)
            assert_adapters_only(model)
            row={'unit':unit,'epoch':exposures/len(fits),'updates_attempted':updates,'image_exposures':exposures,
                 'amp_skipped_total':skipped,'amp_scale':scaler.get_scale(),'train_loss':total_loss/seen,'learning_rate':optimizer.param_groups[0]['lr'],
                 'elapsed_seconds':previous_seconds+time.perf_counter()-tick}
            if vals:
                metrics,_=evaluate(model,pack,vals);row['validation']=metrics
                if metrics['abs_rel']<best:
                    best,best_step=metrics['abs_rel'],updates
                    atomic_save(output/'best.pt',{'model':adapter_state(model),'config':config,'updates':updates,'selection_abs_rel':best})
            history.append(row);write_json(output/'history.json',history)
            atomic_save(output/'resume.pt',{'model':adapter_state(model),'optimizer':optimizer.state_dict(),
                'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'config_hash':config_hash,
                'history':history,'best':best,'best_step':best_step,'unit':unit,'updates':updates,
                'exposures':exposures,'skipped':skipped,'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all()})
            print(json.dumps({'run':job['run_id'],**row}),flush=True)
        atomic_save(output/'final.pt',{'model':adapter_state(model),'config':config,'updates':updates,'selection':'fixed last iterate'})
        if not vals:
            atomic_save(output/'best.pt',{'model':adapter_state(model),'config':config,'updates':updates,'selection':'fixed final refit'})
        selected='best.pt' if job['selection_point']=='best_validation' else 'final.pt'
        load_adapters(model,torch.load(output/selected,map_location='cpu',weights_only=False)['model'])
        measured=None
        if vals:
            _,rows=evaluate(model,pack,vals,detailed=True);measured=save_evaluation(output/'validation',rows)
            expected=best if selected=='best.pt' else history[-1]['validation']['abs_rel']
            assert abs(measured['abs_rel']-expected)<1e-7,('Selected LoRA checkpoint rescoring mismatch',measured['abs_rel'],expected)
            preview(model,pack,vals,output/'visualizations',job['run_id'])
        assert frozen_digest(model)==original_digest,'Original pretrained parameters changed'
        result={'status':'complete','run':job['run_id'],'phase':job['phase'],'smoke':job['smoke'],'family':'lora',
                'job_sha256':record['job_sha256'],'parameters':config['parameters'],'selected_checkpoint':selected,
                'selection_point':job['selection_point'],'best_unit':best_step if vals else None,'completed_units':len(history),
                'updates_attempted':updates,'image_exposures':exposures,'amp_skipped':skipped,'validation':measured,
                'best_checkpoint_sha256':sha(output/'best.pt'),
                'checkpoint_sha256':{name:sha(output/name) for name in ('best.pt','final.pt')},
                'elapsed_seconds':previous_seconds+time.perf_counter()-tick,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                'frozen_parameters_unchanged':True,'training_finished_at':now()}
        write_json(output/'result.json',result);write_json(output/'status.json',{'status':'complete','at':now()})
        append_json(root/study['output']/'runs.jsonl',{'run':job['run_id'],'status':'complete','result':result})
        remove_completed_resume(root/study['output'],output)
        print('PASS '+job['run_id']+' '+json.dumps(measured),flush=True)
        return result
    except BaseException as exc:
        failure={'run':job['run_id'],'status':'failed','at':now(),'error':repr(exc),'traceback':traceback.format_exc()}
        write_json(output/'status.json',failure);append_json(root/study['output']/'runs.jsonl',failure)
        raise
