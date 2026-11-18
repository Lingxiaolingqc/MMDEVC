"""Frozen extension refits and streaming full-test evaluation, with parent controls reused."""
import gc,json,time,sys
from contextlib import redirect_stdout,redirect_stderr
import h5py,numpy as np,torch
from p3_depth.data.nyuv2 import NYUv2Labeled
from p3_depth.residual.records import sha,write_json,now
from p3_depth.residual.final_evaluation import scoring_pack,save_preview
from p3_depth.residual.evaluation import score_image,save_evaluation
from p3_depth.residual.model import corrected_depth
from .common import save_job,snapshot
from .model import make_model
from .data import observable_inputs,load_pack

def check_freeze(root,study):
    path=root/study['output']/'frozen_selection.json';record=json.loads(path.read_text(encoding='utf8'))
    for name,digest in record['source_sha256'].items():assert sha(root/name)==digest,('Source changed after freeze',name)
    for name,digest in record['evidence_sha256'].items():assert sha(root/name)==digest
    assert sha(root/study['split'])==record['split_sha256']
    assert sha(root/study['parent_output']/'frozen_selection.json')==record['parent_freeze_sha256']
    return record,sha(path)

def freeze(root,study,names):
    out=root/study['output'];path=out/'frozen_selection.json'
    assert not path.exists() and not (out/'stage_cache/test/metadata.json').exists()
    decision_path=out/'confirmation_decision.json';decision=json.loads(decision_path.read_text(encoding='utf8'))
    assert sorted(names)==sorted(decision['retained_methods']) and names,'Freeze only confirmed methods'
    review=out/'analysis/development/review.json';assert sha(review)==decision['evidence_sha256']
    evidence={p.relative_to(root).as_posix():sha(p) for p in [decision_path,review]};methods={}
    split=json.loads((root/study['split']).read_text(encoding='utf8'))
    assert sha(root/study['data'])==split['dataset_sha256']
    for name in names:
        runs=[]
        for p in (out/'followups').glob('*/result.json'):
            value=json.loads(p.read_text());cfg=json.loads((p.parent/'config.json').read_text())
            if cfg['candidate']['name']==name and not value['smoke'] and value.get('validation'):
                assert value['validation']['images']==218;runs.append((value,cfg,p))
        assert sorted(cfg['seed'] for _,cfg,_ in runs)==sorted(study['confirmation_seeds'])
        epochs=int(np.median([r['best_unit'] for r,_,_ in runs]))
        methods[name]={'epochs':epochs,'seeds':study['confirmation_seeds'],'development_best_epochs':[r['best_unit'] for r,_,_ in runs]}
        evidence.update({p.relative_to(root).as_posix():sha(p) for _,_,p in runs})
    sources=[*sorted((root/'src/p3_depth/residual_inputs').glob('*.py')),*sorted((root/'src/p3_depth/residual').glob('*.py')),
             root/'configs/residual_inputs_study.json',root/'src/p3_depth/evaluation/course.py',root/'src/p3_depth/evaluation/boundary.py',root/'src/p3_depth/models/runtime.py']
    record={'status':'frozen_before_new_extension_test_inference','recorded_at':now(),'methods':methods,
            'source_sha256':{p.relative_to(root).as_posix():sha(p) for p in sources},'evidence_sha256':evidence,
            'split_sha256':sha(root/study['split']),'protocol':study['protocol'],
            'dataset_sha256':split['dataset_sha256'],'parent_freeze_sha256':sha(root/study['parent_output']/'frozen_selection.json'),
            'duration_rule':'median development-selected epochs; keep80-epoch cosine horizon; final iterate only',
            'controls':'Reuse all three frozen parent S0 tests and B0/B1/B2/B3/R0; no prediction ensemble',
            'test_history':'Parent test and repeated validation already viewed; additional exploratory study, not a new blind test.'}
    write_json(path,record);return record

def refit(root,study):
    from .training import run_job
    record,_=check_freeze(root,study);pack=load_pack(root,study)
    for name,row in record['methods'].items():
        for seed in row['seeds']:
            path=save_job(root,study,name,seed,'refit',row['epochs'])
            logs=root/study['output']/'logs';logs.mkdir(exist_ok=True)
            with (logs/(path.stem+'.log')).open('a',encoding='utf8') as stream:
                with redirect_stdout(Tee(sys.stdout,stream)),redirect_stderr(Tee(sys.stderr,stream)):
                    run_job(root,study,path,True,pack)
            gc.collect();torch.cuda.empty_cache()

class Tee:
    def __init__(self,terminal,stream):self.terminal,self.stream=terminal,stream
    def write(self,value):self.terminal.write(value);self.stream.write(value)
    def flush(self):self.terminal.flush();self.stream.flush()

def evaluate(root,study):
    frozen,digest=check_freeze(root,study);out=root/study['output'];stages=out/'stage_cache/test'
    meta=json.loads((stages/'metadata.json').read_text());assert meta['status']=='complete'
    ids=meta['spec']['ids'];split=json.loads((root/study['split']).read_text());assert ids==split['test_ids'] and len(ids)==654
    source_cache=root/study['parent_output']/'cache/test/predictions.h5'
    assert sha(source_cache)==meta['spec']['parent_cache_sha256']
    stage_arrays={k:np.load(stages/(k+'.npy'),mmap_mode='r') for k in ('init','mid','late')}
    for k in stage_arrays:assert sha(stages/(k+'.npy'))==meta['sha256'][k+'.npy']
    edge=json.loads((root/study['parent_output']/'edge_features/metadata.json').read_text())['calibration']
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with NYUv2Labeled(root/study['data']) as source,h5py.File(source_cache,'r') as cache:
        for name,choice in frozen['methods'].items():
            for seed in choice['seeds']:
                run=f'refit_{name}_seed{seed}';directory=out/'final_test'/run;directory.mkdir(parents=True,exist_ok=True)
                weight=out/'followups'/run/'final.pt';saved=torch.load(weight,map_location='cpu',weights_only=False);config=saved['config'];cfg=config['candidate']
                result=json.loads((weight.parent/'result.json').read_text());assert result['selected_checkpoint']=='final.pt' and sha(weight)==result['checkpoint_sha256']['final.pt']
                spec={'freeze_sha256':digest,'checkpoint_sha256':sha(weight),'stage_metadata_sha256':sha(stages/'metadata.json'),'ids':ids}
                if (directory/'result.json').exists():
                    prior=json.loads((directory/'result.json').read_text());assert prior['spec']==spec
                    assert all(sha(directory/k)==v for k,v in prior['artifact_sha256'].items());continue
                if (directory/'metadata.json').exists():assert json.loads((directory/'metadata.json').read_text())['spec']==spec
                else:write_json(directory/'metadata.json',{'status':'running','spec':spec});snapshot(root,directory)
                first=source.file['images'][ids[0]].transpose(2,1,0).copy()
                inputs=observable_inputs(first,cache[str(ids[0])]['original'][:],cache[str(ids[0])]['confidence'][:],cfg,study['feature_hw'],edge)
                internal={k:stage_arrays[k] for k in cfg['branches']}
                model=make_model(inputs.shape[0],cfg,internal).cuda().eval();model.load_state_dict(saved['model']);rows=[]
                with h5py.File(directory/'predictions.h5','a') as predictions,torch.inference_mode():
                    for i,index in enumerate(ids):
                        key=str(index);rgb=source.file['images'][index].transpose(2,1,0).copy();group=cache[key];base=group['original'][:]
                        if key in predictions and not predictions[key].attrs.get('complete',False):del predictions[key]
                        if key in predictions:prediction=predictions[key][:]
                        else:
                            features=observable_inputs(rgb,base,group['confidence'][:],cfg,study['feature_hw'],edge)
                            x=torch.from_numpy(features[None]).cuda();latents={k:torch.from_numpy(np.array(v[i:i+1],copy=True)).float().cuda() for k,v in internal.items()}
                            with torch.autocast('cuda',dtype=torch.float16):residual=model(x,latents)
                            prediction=corrected_depth(torch.from_numpy(base[None,None]).cuda(),residual,'additive')[0,0].cpu().numpy()
                            d=predictions.create_dataset(key,data=prediction,compression='lzf');predictions.flush();d.attrs['complete']=True;predictions.flush()
                        assert prediction.shape==(480,640) and np.isfinite(prediction).all() and (prediction>0).all()
                        sample=source.load(index+1);rows.append(score_image(prediction,scoring_pack(sample,index),0))
                        if index in (ids[0],ids[len(ids)//2],ids[-1]):save_preview(directory/f'nyuv2_{index+1:04d}.png',rgb,sample.depth_m,base,prediction,sample.valid_gt,run)
                        if (i+1)%100==0 or i+1==len(ids):print('TEST '+run+' '+str(i+1)+'/654',flush=True)
                metrics=save_evaluation(directory,rows)
                write_json(directory/'result.json',{'status':'complete','spec':spec,'method':name,'seed':seed,'metrics':metrics,
                    'artifact_sha256':{k:sha(directory/k) for k in ['predictions.h5','per_image.csv','metrics.json']}})
                write_json(directory/'metadata.json',{'status':'complete','spec':spec});del model;gc.collect();torch.cuda.empty_cache()
