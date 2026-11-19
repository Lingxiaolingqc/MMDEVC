"""Execute immutable follow-up jobs; no test-driven model or duration selection."""
import json
import time
import traceback
import numpy as np
import torch
from p3_depth.residual.records import identity,sha,write_json,append_json,now,remove_completed_resume
from .common import snapshot
from .common import read_job
from .data import fit_job_base,prepare_job_features
from p3_depth.residual.job_schedule import segments,schedule_steps
from .data import load_pack
from .model import make_model
from p3_depth.residual.model import corrected_depth,loss_function
from .evaluation import batch,evaluate,preview
from p3_depth.residual.training import atomic_save
from p3_depth.residual.evaluation import save_evaluation


def run_job(root,study,job_path,resume=False,shared_pack=None):
    record=read_job(root,study,job_path);job=record['job'];cfg=job['candidate'];train=job['training']
    study_root=root/study['output'];output=study_root/'followups'/job['run_id'];output.mkdir(parents=True,exist_ok=True)
    if (output/'result.json').exists():
        result=json.loads((output/'result.json').read_text(encoding='utf-8'))
        if result['job_sha256']!=record['job_sha256'] or any(sha(output/name)!=digest for name,digest in result['checkpoint_sha256'].items()):
            raise ValueError('Completed follow-up identity changed')
        print('PASS already complete '+job['run_id'],flush=True);return result
    if (output/'config.json').exists() and not resume:raise FileExistsError('Existing job requires --resume')
    torch.set_num_threads(4);torch.manual_seed(job['seed']);np.random.seed(job['seed'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    try:
        pack=load_pack(root,study) if shared_pack is None else shared_pack
        positions={index:i for i,index in enumerate(pack['ids'])}
        fits=[positions[index] for index in job['fit_ids']];vals=[positions[index] for index in job['validation_ids']]
        base,calibration=fit_job_base(root,study,job,pack)
        features,internal,input_meta=prepare_job_features(root,study,job,pack,base,calibration)
        model=make_model(features.shape[1],cfg,internal).cuda()
        config={'job':job,'job_sha256':record['job_sha256'],'candidate':cfg,'training':train,'seed':job['seed'],
                'fit_ids':job['fit_ids'],'validation_ids':job['validation_ids'],'feature_hw':job['feature_hw'],
                'calibration':calibration,'cache_sha256':pack['metadata']['cache_sha256'],
                'feature_cache':input_meta,'parameters':sum(p.numel() for p in model.parameters()),
                'amp':True,'gt_in_inference':False,'cudnn_deterministic':True,'deterministic_algorithms_enforced':False}
        config_hash=identity(config)
        if (output/'config.json').exists():
            if identity(json.loads((output/'config.json').read_text(encoding='utf-8')))!=config_hash:raise ValueError('Job config changed')
            if not (output/'provenance.json').exists():
                if (output/'history.json').exists() or (output/'resume.pt').exists():
                    raise ValueError('Training state without provenance; cannot safely resume')
                if (output/'status.json').exists():
                    append_json(output/'pretraining_failures.jsonl',json.loads((output/'status.json').read_text(encoding='utf-8')))
                # A failed source snapshot can leave a config before any update.
                # Preserve its failure and retry the snapshot with the same seed/config.
                snapshot(root,output)
            provenance=json.loads((output/'provenance.json').read_text(encoding='utf-8'))
            for name,digest in provenance['source_sha256'].items():
                if sha(root/name)!=digest:raise ValueError('Source changed; do not resume this job')
        else:write_json(output/'config.json',config);snapshot(root,output)
        optimizer=torch.optim.AdamW(model.parameters(),lr=train['learning_rate'],weight_decay=train['weight_decay'])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,schedule_steps(fits,train),eta_min=train['minimum_learning_rate'])
        scaler=torch.amp.GradScaler('cuda',enabled=True)
        history=[];best_score=float('inf');best_unit=0;stale=0;completed_unit=0;updates=0;exposures=0
        if resume and (output/'resume.pt').exists():
            saved=torch.load(output/'resume.pt',map_location='cpu',weights_only=False)
            if saved['config_hash']!=config_hash:raise ValueError('Resume config mismatch')
            model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler']);scaler.load_state_dict(saved['scaler'])
            history,best_score,best_unit,stale,completed_unit,updates,exposures=[saved[k] for k in ('history','best_score','best_unit','stale','unit','updates','exposures')]
            torch.set_rng_state(saved['rng_cpu']);torch.cuda.set_rng_state_all(saved['rng_cuda'])
        elif resume and (output/'history.json').exists():raise ValueError('History exists without resume checkpoint')
        tick=time.perf_counter();previous_seconds=history[-1]['elapsed_seconds'] if history else 0.
        torch.cuda.reset_peak_memory_stats()
        write_json(output/'status.json',{'status':'running','at':now()})
        append_json(study_root/'runs.jsonl',{'run':job['run_id'],'status':'running','job_sha256':record['job_sha256']})
        if not history:
            initial=evaluate(model,cfg,pack,features,internal,base,vals,train['batch_size'])[0] if vals else {'validation_selection':False,'reason':'frozen all-development refit'}
            write_json(output/'initial_metrics.json',initial)
            print('INITIAL '+job['run_id']+' '+json.dumps(initial),flush=True)
        last_score=next((row['validation']['abs_rel'] for row in reversed(history) if 'validation' in row),None)
        for unit,batches in segments(fits,job['seed'],train):
            if unit<=completed_unit:continue
            model.train();loss_sum=0.;seen=0
            for selected in batches:
                inputs,latent,bases,truths,masks,bands=batch(pack,features,internal,base,selected)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):residual=model(inputs,latent)
                predicted=corrected_depth(bases,residual,cfg['output'])
                loss=loss_function(predicted,truths,masks,cfg['loss'],bands,train['gradient_loss_weight'],train['boundary_loss_weight'])
                if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(predicted).all()):raise ValueError('Nonfinite training result')
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),train['clip_gradient_norm'])
                if not bool(torch.isfinite(norm)):
                    append_json(output/'events.jsonl',{'update':updates+1,'event':'AMP overflow; optimizer step skipped'})
                scaler.step(optimizer);scaler.update();scheduler.step()
                loss_sum+=float(loss.detach())*len(selected);seen+=len(selected);updates+=1;exposures+=len(selected)
            row={'unit':unit,'epoch':exposures/len(fits) if train.get('max_steps') else unit,
                 'updates_attempted':updates,'image_exposures':exposures,'train_loss':loss_sum/seen,
                 'learning_rate':optimizer.param_groups[0]['lr'],'elapsed_seconds':previous_seconds+time.perf_counter()-tick}
            check=bool(train.get('max_steps')) or unit%train['evaluate_every_epochs']==0 or unit==train['max_epochs']
            if vals and check:
                metrics,_=evaluate(model,cfg,pack,features,internal,base,vals,train['batch_size']);row['validation']=metrics
                last_score=metrics['abs_rel']
                if last_score<best_score:
                    best_score,best_unit,stale=last_score,unit,0
                    atomic_save(output/'best.pt',{'model':model.state_dict(),'config':config,'unit':unit,'selection_abs_rel':best_score})
                else:stale+=1
            history.append(row);write_json(output/'history.json',history)
            atomic_save(output/'resume.pt',{'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                        'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'config_hash':config_hash,
                        'history':history,'best_score':best_score,'best_unit':best_unit,'stale':stale,'unit':unit,
                        'updates':updates,'exposures':exposures,'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all()})
            print(json.dumps({'run':job['run_id'],**row}),flush=True)
            if job['selection_point']=='best_validation' and unit>=train['minimum_epochs'] and stale>=train['patience_evaluations']:break
        atomic_save(output/'final.pt',{'model':model.state_dict(),'config':config,'unit':history[-1]['unit'],'selection':'last training iterate'})
        if not vals:
            # Deployment checkpoint is the predeclared final iterate, not a GT-selected best.
            atomic_save(output/'best.pt',{'model':model.state_dict(),'config':config,'unit':history[-1]['unit'],'selection':'fixed final refit'})
        selected='best.pt' if job['selection_point']=='best_validation' else 'final.pt'
        model.load_state_dict(torch.load(output/selected,map_location='cpu',weights_only=False)['model'])
        measured=None;timing=None
        if vals:
            metrics,rows=evaluate(model,cfg,pack,features,internal,base,vals,train['batch_size'],detailed=True)
            measured=save_evaluation(output/'validation',rows);timing=metrics['refiner_seconds_per_image']
            expected=best_score if selected=='best.pt' else last_score
            if abs(measured['abs_rel']-expected)>1e-7:raise ValueError('Selected weight re-score mismatch')
            (output/'visualizations').mkdir(exist_ok=True)
            preview(root,study,output/'visualizations',model,cfg,pack,features,internal,base,vals)
        result={'status':'complete','run':job['run_id'],'phase':job['phase'],'smoke':job['smoke'],'job_sha256':record['job_sha256'],
                'parameters':config['parameters'],'selected_checkpoint':selected,'selection_point':job['selection_point'],
                'best_validation_abs_rel':best_score if vals else None,'best_unit':best_unit if vals else None,
                'completed_units':len(history),'updates_attempted':updates,'image_exposures':exposures,
                'validation':measured,'best_checkpoint_sha256':sha(output/'best.pt'),
                'checkpoint_sha256':{name:sha(output/name) for name in ('best.pt','final.pt')},
                'elapsed_seconds':previous_seconds+time.perf_counter()-tick,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                'refiner_seconds_per_image':timing,'timing_scope':'refiner GPU forward and restore; excludes cached foundation/input preparation',
                'training_finished_at':now()}
        write_json(output/'result.json',result);write_json(output/'status.json',{'status':'complete','at':now()})
        append_json(study_root/'runs.jsonl',{'run':job['run_id'],'status':'complete','result':result})
        remove_completed_resume(study_root,output)
        print('PASS '+job['run_id']+' '+json.dumps(measured),flush=True)
        return result
    except BaseException as exc:
        failed={'run':job['run_id'],'status':'failed','at':now(),'error':repr(exc),'traceback':traceback.format_exc()}
        write_json(output/'status.json',failed);append_json(study_root/'runs.jsonl',failed);raise
