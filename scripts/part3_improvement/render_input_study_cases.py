"""Matched RGB, GT, base, S0 and candidate views with fixed colour scales."""
import argparse,json,sys
from pathlib import Path
import h5py,numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_residual_study import read_csv
from benchmark_input_study import load_refiner
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from p3_depth.data.nyuv2 import NYUv2Labeled
from p3_depth.residual.deployment import image_inputs,predict_arrays
from p3_depth.residual.model import corrected_depth
from p3_depth.residual.records import write_json,sha
from p3_depth.residual_inputs.common import load
from p3_depth.residual_inputs.data import observable_inputs
from p3_depth.residual.final_evaluation import scoring_pack
from p3_depth.residual.evaluation import score_image
from p3_depth.evaluation.boundary import boundary_masks

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,help='Completed extension run directory, relative to repository')
    a=p.parse_args();s,_=load(ROOT);out=ROOT/s['output'];old=ROOT/s['parent_output'];run=(ROOT/a.run).resolve()
    assert run.is_relative_to(out.resolve())
    result=json.loads((run/'result.json').read_text());cfg=json.loads((run/'config.json').read_text())
    assert result['status']=='complete' and not result['smoke'] and result['validation']['images']==218
    seed=cfg['seed'];ref=old/'runs'/f'frozen_features_seed{seed}'
    before={int(x['source_index']):x for x in read_csv(ref/'validation/per_image.csv')}
    after={int(x['source_index']):x for x in read_csv(run/'validation/per_image.csv')}
    assert before.keys()==after.keys()
    ids=list(after);ranked=sorted(ids,key=lambda i:float(after[i]['abs_rel'])-float(before[i]['abs_rel']))
    choices={'fixed_first':ids[0],'fixed_middle':ids[len(ids)//2],'fixed_last':ids[-1],
             'largest_gain':ranked[0],'median_change':ranked[len(ranked)//2],'largest_regression':ranked[-1],
             'largest_boundary_regression':min(ids,key=lambda i:float(after[i]['boundary_f1'])-float(before[i]['boundary_f1']))}
    directory=out/'analysis/visuals'/run.name;directory.mkdir(parents=True,exist_ok=True)
    stage=out/'stage_cache/development';meta=json.loads((stage/'metadata.json').read_text());positions={v:i for i,v in enumerate(meta['spec']['ids'])}
    arrays={k:np.load(stage/(k+'.npy'),mmap_mode='r') for k in cfg['candidate']['branches']}
    edge=json.loads((old/'edge_features/metadata.json').read_text())['calibration']
    original_model,original_cfg=load_refiner(ref/'best.pt');model,model_cfg=load_refiner(run/'best.pt')
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    checks=[]
    with NYUv2Labeled(ROOT/s['data']) as source,h5py.File(old/'cache/development/predictions.h5','r') as cache,torch.inference_mode():
        for reason,index in choices.items():
            sample=source.load(index+1);rgb=sample.rgb;group=cache[str(index)];base=group['original'][:]
            x=image_inputs(rgb,base,original_cfg['candidate'],original_cfg['feature_hw'],original_cfg['calibration'],internal=group['internal'][:])
            reference=predict_arrays(original_model,original_cfg,*x)[0].cpu().numpy()
            features=observable_inputs(rgb,base,group['confidence'][:],cfg['candidate'],s['feature_hw'],edge)
            pos=positions[index];latents={k:torch.from_numpy(np.array(v[pos:pos+1],copy=True)).float().cuda() for k,v in arrays.items()}
            with torch.autocast('cuda',dtype=torch.float16):residual=model(torch.from_numpy(features[None]).cuda(),latents)
            prediction=corrected_depth(torch.from_numpy(base[None,None]).cuda(),residual,'additive')[0,0].cpu().numpy()
            pack=scoring_pack(sample,index);measured=[score_image(v,pack,0) for v in [reference,prediction]]
            differences=[abs(measured[j]['abs_rel']-float(row['abs_rel'])) for j,row in enumerate([before[index],after[index]])]
            # Batch1 versus the saved validation batch may have tiny FP16 roundoff.
            assert max(differences)<1e-4,('Saved-score reproduction differs',index,differences)
            truth=sample.depth_m;mask=pack['mask'][0]
            fig,axes=plt.subplots(3,5,figsize=(17,10),constrained_layout=True)
            axes[0,0].imshow(rgb);axes[0,0].set_title('RGB')
            for ax,value,label in zip(axes[0,1:],[truth,base,reference,prediction],['GT','UniDepthV2','Previous S0',cfg['candidate']['name']]):
                ax.imshow(value,vmin=0,vmax=10,cmap='viridis');ax.set_title(label+' | 0-10 m')
            axes[1,0].imshow(mask,cmap='gray');axes[1,0].set_title('Fixed course evaluation mask')
            axes[1,1].axis('off');axes[1,1].text(.05,.6,f'{reason}\nnyuv2_{index+1:04d}\nS0 AbsRel: {measured[0]["abs_rel"]:.5f}\nNew AbsRel: {measured[1]["abs_rel"]:.5f}',fontsize=12)
            for ax,value in zip(axes[1,2:],[base,reference,prediction]):
                error=np.where(mask,np.abs(np.clip(value,.001,10)-truth)/np.maximum(truth,.001),np.nan)
                artist=ax.imshow(error,vmin=0,vmax=.5,cmap='magma');ax.set_title('Absolute relative error | 0-0.5')
            axes[2,0].text(.05,.4,'Depth edges:\nGT = cyan\nPrediction = magenta\nSame pixel = white\n\nScores allow a 2px tolerance',fontsize=11)
            _,_,gt_edges,_=boundary_masks(truth,truth,mask,mask)
            axes[2,1].imshow(gt_edges,cmap='gray');axes[2,1].set_title('GT depth edges')
            for ax,value in zip(axes[2,2:],[base,reference,prediction]):
                clipped=np.clip(value,.001,10).astype(np.float32)
                _,_,gt_edge,pred_edge=boundary_masks(clipped,truth,mask,np.isfinite(clipped)&(clipped>0))
                overlay=np.repeat((rgb.mean(-1)/255*.25)[...,None],3,-1)
                overlay[gt_edge]=[0,1,1];overlay[pred_edge]=[1,0,1];overlay[gt_edge&pred_edge]=[1,1,1]
                ax.imshow(overlay);ax.set_title('GT / predicted depth edges')
            for ax in axes.flat:ax.axis('off')
            fig.colorbar(artist,ax=axes[1,2:],shrink=.65)
            fig.suptitle('Validation diagnostic: matched samples, fixed scales; selected examples do not represent the full distribution')
            name=f'{reason}_nyuv2_{index+1:04d}.png';fig.savefig(directory/name,dpi=120);plt.close(fig)
            checks.append({'reason':reason,'source_index':index,'s0_absrel':measured[0]['abs_rel'],'candidate_absrel':measured[1]['abs_rel'],
                           's0_boundary_f1':measured[0]['boundary_f1'],'candidate_boundary_f1':measured[1]['boundary_f1'],
                           'saved_score_abs_difference':differences,'figure':name,'sha256':sha(directory/name)})
    write_json(directory/'cases.json',{'status':'complete','scope':'218-image validation; three fixed cases, largest gain/median change/largest regression by paired AbsRel, plus largest F1 regression; extreme examples do not represent mean quality',
        'checkpoints':{str(path.relative_to(ROOT)):sha(path) for path in [ref/'best.pt',run/'best.pt']},'cases':checks})
    print('PASS matched validation cases: '+str(directory),flush=True)

if __name__=='__main__':main()
