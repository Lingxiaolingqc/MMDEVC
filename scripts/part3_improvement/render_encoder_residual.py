"""Fixed validation cases and same-session cached-refiner costs for completed pairs."""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from p3_depth.encoder_residual.common import load,check_freeze,reference_path
from p3_depth.encoder_residual.model import make_model
from p3_depth.stylesketch_residual.model import StyleSketchResidual
from p3_depth.stylesketch_residual.data import Capture as DecoderCapture
from p3_depth.models.runtime import ImageModel
from p3_depth.data.nyuv2 import NYUv2Labeled
from p3_depth.adaptation_study.common import candidate as old_candidate
from p3_depth.adaptation_study.model import make_model as old_model
from p3_depth.residual_inputs.data import observable_inputs
from p3_depth.residual.model import corrected_depth
from p3_depth.residual.final_evaluation import scoring_pack
from p3_depth.residual.evaluation import score_image
from p3_depth.evaluation.boundary import boundary_masks
from p3_depth.residual.records import sha,write_json


def forward(model,inputs,latent,base):
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16): residual=model(inputs,latent)
    return corrected_depth(base,residual,'additive')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--methods',nargs='+');args=p.parse_args()
    study=load(ROOT);check_freeze(ROOT,study);out=ROOT/study['output'];dest=out/'analysis/visualizations';dest.mkdir(parents=True,exist_ok=True)
    methods=args.methods or [name for name in study['queue'] if (out/'runs'/f'{name}_seed320114/result.json').exists()]
    assert methods and set(methods)<=set(study['queue'])
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    settings=yaml.safe_load((ROOT/'configs/paths.local.yaml').read_text())
    adapter=ImageModel('unidepth',ROOT,settings,'cuda',precision='fp16',threads=4)
    adapter.model.requires_grad_(False).eval(); decoder=DecoderCapture(adapter.model); encoded={}
    def hook(module,args,output):
        for block in study['encoder_blocks']:encoded[f'b{block:02d}']=output[0][block-1].permute(0,3,1,2).detach()
    handle=adapter.model.pixel_encoder.register_forward_hook(hook)
    old_study=json.loads((ROOT/'configs/fusion_lora_study.json').read_text())
    ss_study=json.loads((ROOT/'configs/stylesketch_residual.json').read_text())
    fm=json.loads((out/'features/metadata.json').read_text()); positions={index:i for i,index in enumerate(fm['ids'])}
    checkpoints={};scores=[];bench=[];previews=[];parities=[]
    with NYUv2Labeled(ROOT/study['data']) as data,torch.inference_mode():
        for sample_id in study['fixed_visual_sample_ids']:
            sample=data.load(sample_id); index=sample_id-1; pos=positions[index]
            assert index in json.loads((ROOT/study['split']).read_text())['validation_ids']
            result=adapter.model.infer(torch.from_numpy(sample.rgb.copy()).permute(2,0,1))
            base=result['depth'].float()
            # Reproduce cached FP16 features before converting back to float32.
            enc={k:v.half().float() for k,v in encoded.items()}
            dec={'init':result['depth_features'].half().float(),**{k:v.half().float() for k,v in decoder.values.items()}}
            cached=np.load(ROOT/study['disk_inputs']/'original.npy',mmap_mode='r')
            assert np.array_equal(base[0,0].cpu().numpy(),cached[pos]);cached._mmap.close()
            for name in enc:
                cached=np.load(out/'features'/(name+'.npy'),mmap_mode='r')
                assert np.array_equal(enc[name][0].half().cpu().numpy(),cached[pos]);cached._mmap.close()
            parities.append({'sample_id':sample_id,'foundation_depth_and_encoder_cache_exact':True})
            inputs=torch.from_numpy(observable_inputs(sample.rgb,base[0,0].cpu().numpy(),None,{'drop_rgb':False,'extra_inputs':[]},study['feature_hw'],{}))[None].cuda()
            pack=scoring_pack(sample,index)
            for name in methods:
                cfg=study['candidates'][name]; reference=cfg['reference']
                oldcfg=ss_study['candidate'] if reference=='SS1' else old_candidate(old_study,reference)
                latent={alias:enc[source] for alias,source in cfg['source_map'].items()}
                left=StyleSketchResidual(oldcfg).cuda() if reference=='SS1' else old_model(4,oldcfg,dec).cuda()
                right=make_model(4,cfg,latent).cuda()
                old_path=reference_path(ROOT,study,reference,320112)/'best.pt'
                new_path=out/'runs'/f'{name}_seed320112'/'best.pt'
                left.load_state_dict(torch.load(old_path,map_location='cpu',weights_only=False)['model'])
                right.load_state_dict(torch.load(new_path,map_location='cpu',weights_only=False)['model'])
                left.eval();right.eval()
                a=forward(left,inputs,dec,base)[0,0].cpu().numpy()
                b=forward(right,inputs,latent,base)[0,0].cpu().numpy()
                checkpoints[reference]=sha(old_path);checkpoints[name]=sha(new_path)
                values=[base[0,0].cpu().numpy(),a,b]; labels=['B0',reference,name]
                for label,array in zip(labels,values):
                    scores.append({'comparison':name,'method':label,'sample_id':sample_id,**score_image(array,pack,0)})
                fig,axes=plt.subplots(3,4,figsize=(13,8.5),constrained_layout=True)
                axes[0,0].imshow(sample.rgb);axes[0,0].set_title('RGB')
                axes[1,0].imshow(sample.depth_m,vmin=0,vmax=10,cmap='viridis');axes[1,0].set_title('GT depth (0-10 m)')
                _,_,gt_edges,_=boundary_masks(sample.depth_m,sample.depth_m,sample.valid_gt,sample.valid_gt)
                axes[2,0].imshow(gt_edges,cmap='gray');axes[2,0].set_title('GT depth boundaries')
                for col,(label,array) in enumerate(zip(labels,values),1):
                    clipped=np.clip(array,.001,10)
                    axes[0,col].imshow(clipped,vmin=0,vmax=10,cmap='viridis');axes[0,col].set_title(label+' depth')
                    error=np.ma.array(np.abs(clipped-sample.depth_m)/np.maximum(sample.depth_m,.001),mask=~sample.valid_gt)
                    axes[1,col].imshow(error,vmin=0,vmax=.5,cmap='magma');axes[1,col].set_title('Relative error (0-0.5)')
                    _,_,_,edges=boundary_masks(clipped,sample.depth_m,sample.valid_gt,sample.valid_gt)
                    axes[2,col].imshow(edges,cmap='gray');axes[2,col].set_title('Predicted boundaries')
                for ax in axes.flat: ax.axis('off')
                fig.suptitle(f'nyuv2_{sample_id:04d}; {sample.scene}; seed320112; fixed limits; batch-1 qualitative replay')
                target=dest/f'{name}_nyuv2_{sample_id:04d}.png';fig.savefig(target,dpi=145);plt.close(fig)
                previews.append(target.relative_to(out).as_posix())
                if sample_id==study['fixed_visual_sample_ids'][0]:
                    for label,model,features in [(reference,left,dec),(name,right,latent)]:
                        for _ in range(10):forward(model,inputs,features,base)
                        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();resident=torch.cuda.memory_allocated()/2**20
                        times=[]
                        for _ in range(30):
                            torch.cuda.synchronize();t=time.perf_counter();forward(model,inputs,features,base);torch.cuda.synchronize();times.append(time.perf_counter()-t)
                        bench.append({'method':label,'comparison':name,'batch':1,'sample_id':sample_id,'warmups':10,'repeats':30,
                            'median_ms':float(np.median(times)*1000),'p95_ms':float(np.quantile(times,.95)*1000),
                            'resident_allocated_mib':resident,'peak_incremental_mib':torch.cuda.max_memory_allocated()/2**20-resident,
                            'scope':'Cached refiner forward and original-grid restore; excludes foundation, transfer and rendering. Foundation and pair remain resident.'})
                del left,right
            print('FIXED CASE '+str(sample_id),flush=True)
    decoder.close();handle.remove()
    write_json(out/'analysis/visual_review_data.json',{'seed':320112,'fixed_sample_ids':study['fixed_visual_sample_ids'],
        'batch_size':1,'batch_note':'Qualitative replay uses batch1; formal 218-image results retain original batch8 scoring.',
        'scores':scores,'previews':previews,'parities':parities,'checkpoint_sha256':checkpoints,'benchmark':bench,
        'source_sha256':sha(Path(__file__))})
    print('PASS fixed cases and same-session paired cached-refiner benchmark',flush=True)


if __name__=='__main__':main()
