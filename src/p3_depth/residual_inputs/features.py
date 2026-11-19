"""Read-only hooks on official decoder stages; no ground truth enters inference."""
import json,time
import h5py,numpy as np,torch,yaml
from p3_depth.models.runtime import ImageModel
from p3_depth.data.nyuv2 import NYUv2Labeled
from p3_depth.residual.records import sha,write_json,identity
from .common import snapshot

class StageCapture:
    def __init__(self,model,names=('mid','late')):
        self.values={};layer=model.pixel_decoder.depth_layer
        indices={'mid':-2,'late':-1};assert set(names)<=set(indices)
        self.handles=[layer.ups[indices[name]].register_forward_hook(self.hook(name)) for name in names]
    def hook(self,name):
        def save(module,args,output):self.values[name]=output.detach()
        return save
    def close(self):
        for h in self.handles:h.remove()

def prepare(root,study,parent,split_name='development',limit=None):
    directory=root/study['output']/'stage_cache'/(split_name+(f'_smoke{limit}' if limit else ''))
    directory.mkdir(parents=True,exist_ok=True)
    prior=root/parent['output']/'cache'/split_name
    meta=json.loads((prior/'metadata.json').read_text(encoding='utf8'))
    assert meta['status']=='complete' and sha(prior/'predictions.h5')==meta['cache_sha256']
    ids=meta['config']['ids'][:limit] if limit else meta['config']['ids']
    if split_name=='test':assert (root/study['output']/'frozen_selection.json').exists()
    spec={'ids':ids,'parent_cache_sha256':meta['cache_sha256'],'stage_modules':['depth_layer.ups[-2]','depth_layer.ups[-1]'],
          'grid':'native; project then bilinear to240x320; no padding supported in this fixed NYU study',
          'dtype':'float16','gt_used_in_features':False}
    target=directory/'metadata.json'
    if target.exists():
        previous=json.loads(target.read_text(encoding='utf8'))
        assert previous['spec']==spec
        if previous['status']=='complete':
            assert all(sha(directory/n)==v for n,v in previous['sha256'].items());print('PASS reused stages '+str(directory),flush=True);return previous
    settings=yaml.safe_load((root/'configs/paths.local.yaml').read_text())
    adapter=ImageModel('unidepth',root,settings,'cuda',precision='fp16',threads=4)
    adapter.model.requires_grad_(False).eval()
    from unidepth.models.unidepthv2.unidepthv2 import get_paddings
    # infer uses model.shape_constraints; the cache records the actual check.
    constraints=adapter.model.shape_constraints
    pads,shape=get_paddings((480,640),constraints['ratio_bounds'])
    assert not any(pads),'Native features must be unpadded before use; do not change reference silently'
    capture=StageCapture(adapter.model);arrays={};diffs=[];tick=time.perf_counter();snapshot(root,directory)
    write_json(target,{'status':'running','spec':spec})
    with NYUv2Labeled(root/study['data']) as source,h5py.File(prior/'predictions.h5','r') as old:
        for n,index in enumerate(ids):
            rgb=source.file['images'][index].transpose(2,1,0).copy();tensor=torch.from_numpy(rgb).permute(2,0,1)
            with torch.inference_mode():result=adapter.model.infer(tensor)
            prediction=result['depth'][0,0].float().cpu().numpy()
            delta=float(np.max(np.abs(prediction-old[str(index)]['original'][:])))
            assert delta<=1e-5,('Foundation prediction changed',index,delta)
            values={'init':result['depth_features'][0].float().cpu().numpy(),
                    **{k:v[0].float().cpu().numpy() for k,v in capture.values.items()}}
            assert np.array_equal(values['init'].astype(np.float16),old[str(index)]['internal'][:]),'Initial feature identity changed'
            diffs.append(delta)
            for name,value in values.items():
                assert np.isfinite(value).all()
                if name not in arrays:
                    arrays[name]=np.lib.format.open_memmap(directory/(name+'.npy'),mode='w+',dtype=np.float16,shape=(len(ids),*value.shape))
                arrays[name][n]=value
            if (n+1)%50==0 or n==0 or n+1==len(ids):print('STAGES '+str(n+1)+'/'+str(len(ids))+' '+str({k:v.shape for k,v in values.items()}),flush=True)
        for array in arrays.values():array.flush()
    capture.close()
    result={'status':'complete','spec':spec,'paddings':list(pads),'padded_hw':shape,'model':adapter.metadata,
            'shapes':{k:list(v.shape) for k,v in arrays.items()},'parent_depth_max_abs_difference':max(diffs),
            'init_bitwise_equal_all_images':True,'seconds':time.perf_counter()-tick,
            'sha256':{name+'.npy':sha(directory/(name+'.npy')) for name in arrays}}
    write_json(target,result);print('PASS stages '+str(directory),flush=True);return result
