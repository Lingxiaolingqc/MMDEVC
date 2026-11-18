"""Full official test evaluation from frozen specifications and final weights."""
import gc
import json
import time
import h5py
import numpy as np
import torch
from p3_depth.data.nyuv2 import NYUv2Labeled
from p3_depth.evaluation.boundary import boundary_masks
from p3_depth.evaluation.rgb_depth_edges import near_edges
from .data import load_pack,fit_scale
from .deployment import image_inputs,load_refiner,predict_arrays,needs_edges
from .evaluation import score_image,save_evaluation
from .records import sha,identity,write_json,snapshot,now
from .selection import load_selection,check_refit_spec


def final_calibration(root,study,frozen):
    output=root/study['output'];path=output/'final_calibration.json'
    dev_meta=json.loads((output/'cache/development/metadata.json').read_text(encoding='utf-8'))
    spec={'fit_ids':frozen['development_ids'],'cache_sha256':dev_meta['cache_sha256'],'dataset_sha256':frozen['dataset_sha256']}
    if path.exists():
        result=json.loads(path.read_text(encoding='utf-8'))
        if result['spec']!=spec:raise ValueError('Final development calibration changed')
        return result
    pack=load_pack(root,study);positions=list(range(len(pack['ids'])))
    if set(pack['ids'])!=set(frozen['development_ids']):raise ValueError('Final calibration is not all development data')
    result={'spec':spec}
    for name,base in [('scale',pack['original']),('flip_scale',.75*pack['original']+.25*pack['flip'])]:
        result[name]=fit_scale(base,pack['truth'],pack['mask'],positions)
    write_json(path,result);del pack;gc.collect()
    return result


def scoring_pack(sample,index):
    domain,_,edges,_=boundary_masks(sample.depth_m,sample.depth_m,sample.valid_gt,sample.valid_gt)
    return {'ids':[index],'scenes':[sample.scene],'truth':sample.depth_m[None],'mask':sample.valid_gt[None],
            'band':(near_edges(edges,2)&domain)[None]}


def save_preview(path,rgb,truth,base,prediction,mask,label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,5,figsize=(15,3.5),constrained_layout=True)
    axes[0].imshow(rgb)
    for ax,depth in zip(axes[1:4],[truth,base,prediction]):ax.imshow(depth,vmin=0,vmax=10,cmap='viridis')
    error=np.where(mask,np.abs(np.clip(prediction,.001,10)-truth),np.nan)
    axes[4].imshow(error,vmin=0,vmax=1,cmap='magma')
    for ax,title in zip(axes,['RGB','GT (0-10 m)','Base (0-10 m)','Output (0-10 m)','Abs error (0-1 m)']):
        ax.set_title(title);ax.axis('off')
    fig.suptitle(label+' | official test; fixed display limits');fig.savefig(path,dpi=140);plt.close(fig)


def completed_refits(root,study,frozen,frozen_hash):
    entries=[]
    for planned in frozen['refits']:
        directory=root/study['output']/'followups'/planned['run_id']
        result=json.loads((directory/'result.json').read_text(encoding='utf-8'))
        config=json.loads((directory/'config.json').read_text(encoding='utf-8'))
        check_refit_spec(config['job'],frozen)
        if result['status']!='complete' or result['phase']!='refit' or result['selected_checkpoint']!='final.pt':
            raise ValueError('Final refit is incomplete or did not select the fixed final iterate')
        if config['job']['frozen_selection_sha256']!=frozen_hash:raise ValueError('Refit used another freeze')
        checkpoint=directory/'final.pt'
        if sha(checkpoint)!=result['checkpoint_sha256']['final.pt']:raise ValueError('Final weights changed')
        entries.append((planned,checkpoint,config))
    return entries


def evaluate_final(root,study):
    frozen,frozen_hash=load_selection(root,study)
    refits=completed_refits(root,study,frozen,frozen_hash)
    calibration=final_calibration(root,study,frozen)
    output=root/study['output'];cache_path=output/'cache/test/predictions.h5'
    cache_meta=json.loads((cache_path.parent/'metadata.json').read_text(encoding='utf-8'))
    development_meta=json.loads((output/'cache/development/metadata.json').read_text(encoding='utf-8'))
    if cache_meta['status']!='complete' or sha(cache_path)!=cache_meta['cache_sha256']:
        raise ValueError('Test predictions incomplete or changed')
    if cache_meta['config']['ids']!=frozen['test_ids'] or sha(root/study['data'])!=frozen['dataset_sha256']:
        raise ValueError('Test/data identities differ from freeze')
    for key in ('model','asset','requested_precision','input_size','camera_input','gt_used_by_model'):
        if cache_meta['model'][key]!=development_meta['model'][key]:raise ValueError('Foundation inference changed between development/test')
    configurations=[(name,None,{'candidate':{'base':base},'calibration':calibration}) for name,base in [('B0','original'),('B1','scale'),('B2','flip'),('B3','flip_scale')]]
    configurations += [(row['run_id'],path,config) for row,path,config in refits]
    ids=frozen['test_ids'];fixed_previews={ids[0],ids[len(ids)//2],ids[-1]};summaries={}
    with h5py.File(cache_path,'r') as cache,NYUv2Labeled(root/study['data']) as source:
        for label,checkpoint,config in configurations:
            cfg=config['candidate'];directory=output/'final_test'/label;directory.mkdir(parents=True,exist_ok=True)
            spec={'frozen_sha256':frozen_hash,'cache_sha256':cache_meta['cache_sha256'],'config_sha256':identity(config),
                  'checkpoint_sha256':sha(checkpoint) if checkpoint else None,'ids':ids,'protocol':frozen['protocol']}
            result_path=directory/'result.json';prediction_path=directory/'predictions.h5'
            if result_path.exists():
                previous=json.loads(result_path.read_text(encoding='utf-8'))
                if previous['spec']!=spec or any(sha(directory/name)!=value for name,value in previous['artifact_sha256'].items()):
                    raise ValueError('Completed final evaluation changed')
                summaries[label]=previous['metrics'];print('PASS reused final '+label,flush=True);continue
            meta_path=directory/'metadata.json'
            if meta_path.exists() and json.loads(meta_path.read_text(encoding='utf-8'))['spec']!=spec:
                raise ValueError('Cannot resume test evaluation with different methods/settings')
            write_json(meta_path,{'status':'running','spec':spec,'at':now()});snapshot(root,directory)
            model=None;edge_cal=None
            if checkpoint:
                model,loaded_config=load_refiner(checkpoint)
                if identity(loaded_config)!=identity(config):raise ValueError('Weight/config mismatch')
                if needs_edges(cfg):edge_cal=config['feature_cache']['spec']['edge_calibration']
            rows=[];timings=[];tick=time.perf_counter()
            with h5py.File(prediction_path,'a') as predictions:
                for n,index in enumerate(ids):
                    group=cache[str(index)];original=group['original'][:];flipped=group['flip'][:]
                    rgb=source.file['images'][index].transpose(2,1,0).copy()
                    if cfg['base'] in ('original','scale'):base=original
                    else:base=.75*original+.25*flipped
                    if cfg['base'] in ('scale','flip_scale'):base=base*config['calibration'][cfg['base']]['scale']
                    key=str(index)
                    if key in predictions and not predictions[key].attrs.get('complete',False):
                        del predictions[key]
                    if key in predictions:
                        prediction=predictions[key][:]
                        measured_time=float(predictions[key].attrs.get('refiner_seconds',0.))
                    else:
                        if model is None:prediction=base;measured_time=0.
                        else:
                            arrays=image_inputs(rgb,original,cfg,config['feature_hw'],config['calibration'],
                                flipped=flipped,confidence=group['confidence'][:],
                                internal=group['internal'][:] if cfg['inputs']=='frozen_features' else None,edge_calibration=edge_cal)
                            torch.cuda.synchronize();start=time.perf_counter()
                            prediction=predict_arrays(model,config,*arrays)[0].float().cpu().numpy()
                            torch.cuda.synchronize();measured_time=time.perf_counter()-start
                        if prediction.shape!=(480,640) or not np.isfinite(prediction).all() or (prediction<=0).any():
                            raise ValueError('Invalid final prediction; do not omit pixels')
                        saved=predictions.create_dataset(key,data=prediction.astype(np.float32),compression='lzf')
                        saved.attrs['refiner_seconds']=measured_time
                        predictions.flush();saved.attrs['complete']=True;predictions.flush()
                    if prediction.shape!=(480,640) or not np.isfinite(prediction).all() or (prediction<=0).any():
                        raise ValueError('Invalid saved final prediction')
                    timings.append(measured_time)
                    sample=source.load(index+1);pack=scoring_pack(sample,index);rows.append(score_image(prediction,pack,0))
                    if index in fixed_previews:
                        save_preview(directory/f'nyuv2_{index+1:04d}.png',rgb,sample.depth_m,base,prediction,sample.valid_gt,label)
                    if (n+1)%100==0 or n+1==len(ids):print(f'TEST {label} {n+1}/{len(ids)}',flush=True)
            metrics=save_evaluation(directory,rows)
            result={'status':'complete','spec':spec,'metrics':metrics,'at':now(),'test_history':frozen['test_history'],
                    'elapsed_seconds':time.perf_counter()-tick,'mean_refiner_seconds_with_transfer':float(np.mean(timings)),
                    'timing_scope':'batch1 refiner plus CPU/GPU transfer; cached foundation and CPU input formatting excluded; not an isolated end-to-end benchmark',
                    'artifact_sha256':{name:sha(directory/name) for name in ('predictions.h5','per_image.csv','metrics.json')}}
            write_json(result_path,result);write_json(meta_path,{'status':'complete','spec':spec,'at':now()})
            summaries[label]=metrics
            del model;gc.collect();torch.cuda.empty_cache()
            print('PASS final '+label+' '+json.dumps(metrics),flush=True)
    write_json(output/'final_test/comparison.json',{'frozen_sha256':frozen_hash,'metrics':summaries,'test_history':frozen['test_history']})
