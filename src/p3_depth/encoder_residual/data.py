"""Read-only genuine encoder captures and bounded-resident verified data access."""
import json
import time
import h5py
import numpy as np
import torch
import yaml
from p3_depth.models.runtime import ImageModel
from p3_depth.data.nyuv2 import NYUv2Labeled
from p3_depth.stylesketch_residual.data import state_digest
from p3_depth.stylesketch_residual.low_memory import DiskArray, verify
from p3_depth.residual.records import sha, write_json


def prepare(root, study):
    out = root/study['output']/'features'; out.mkdir(parents=True,exist_ok=True)
    prior = root/study['parent_output']/'cache/development'
    meta = json.loads((prior/'metadata.json').read_text())
    split = json.loads((root/study['split']).read_text())
    verify(root/study['data'],split['dataset_sha256'])
    verify(prior/'predictions.h5',meta['cache_sha256'])
    ids = meta['config']['ids']
    assert ids == sorted(split['fit_ids']+split['validation_ids']) and len(ids)==795
    assert not set(ids)&set(split['test_ids'])
    if (out/'metadata.json').exists():
        saved = json.loads((out/'metadata.json').read_text())
        assert saved['status']=='complete', 'Preserve incomplete cache and inspect failure'
        assert saved['blocks']==study['encoder_blocks'] and saved['ids']==ids
        for name,digest in saved['sha256'].items(): verify(out/name,digest)
        return saved
    settings = yaml.safe_load((root/'configs/paths.local.yaml').read_text())
    adapter = ImageModel('unidepth',root,settings,'cuda',precision='fp16',threads=4)
    model = adapter.model.eval().requires_grad_(False)
    assert not any(p.requires_grad for p in model.parameters())
    assert not any(m.training for m in model.modules())
    before = state_digest(model); captured={}
    encoder = model.pixel_encoder
    assert len(encoder.blocks)==12 and encoder.embed_dim==384
    def receive(module,args,output):
        assert len(output[0])==12
        for block in study['encoder_blocks']:
            captured[f'b{block:02d}']=output[0][block-1].detach().permute(0,3,1,2)
    handle=encoder.register_forward_hook(receive)
    from unidepth.models.unidepthv2.unidepthv2 import get_paddings
    pads,_=get_paddings((480,640),model.shape_constraints['ratio_bounds'])
    assert not any(pads)
    write_json(out/'metadata.json',{'status':'running','ids':ids,'blocks':study['encoder_blocks']})
    arrays={}; max_difference=0.; distinct_min=float('inf'); tick=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with NYUv2Labeled(root/study['data']) as source,h5py.File(prior/'predictions.h5','r') as old:
        for n,index in enumerate(ids):
            rgb=source.file['images'][index].transpose(2,1,0).copy()
            with torch.inference_mode(): result=model.infer(torch.from_numpy(rgb).permute(2,0,1))
            difference=float(np.max(np.abs(result['depth'][0,0].float().cpu().numpy()-old[str(index)]['original'][:])))
            assert difference<=1e-5,(index,difference)
            assert np.array_equal(result['depth_features'][0].half().cpu().numpy(),old[str(index)]['internal'][:])
            max_difference=max(max_difference,difference)
            previous=None
            for name,tensor in captured.items():
                value=tensor[0].float().cpu().numpy()
                assert value.shape==(384,31,41) and np.isfinite(value).all(),(name,value.shape)
                if previous is not None:
                    delta=float(np.max(np.abs(value-previous))); assert delta>0
                    distinct_min=min(distinct_min,delta)
                previous=value
                if name not in arrays:
                    arrays[name]=np.lib.format.open_memmap(out/(name+'.npy'),mode='w+',dtype=np.float16,shape=(len(ids),*value.shape))
                arrays[name][n]=value
                assert np.isfinite(arrays[name][n]).all()
            if n==0 or (n+1)%50==0 or n+1==len(ids):
                print(f'ENCODER {n+1}/{len(ids)}: seven distinct native block outputs; D0 parity {max_difference}',flush=True)
    handle.remove()
    shapes={k:list(a.shape) for k,a in arrays.items()}
    for a in arrays.values(): a.flush(); a._mmap.close()
    after=state_digest(model)
    assert before==after and all(p.grad is None for p in model.parameters())
    record={'status':'complete','ids':ids,'blocks':study['encoder_blocks'],'shapes':shapes,
        'source':'pixel_encoder.forward outputs[block-1]; native NHWC->NCHW, after configured normalization/token removal, before stacking',
        'official_use_norm':encoder.use_norm,'parent_cache_sha256':meta['cache_sha256'],
        'split_sha256':sha(root/study['split']),'dataset_sha256':split['dataset_sha256'],
        'model':adapter.metadata,'foundation_state_before':before,'foundation_state_after':after,
        'foundation_frozen':True,'gt_used_in_features':False,'paddings':list(pads),
        'original_depth_max_abs_difference':max_difference,'decoder_init_fp16_bitwise_equal_all_images':True,
        'distinct_feature_max_difference_minimum':distinct_min,'elapsed_seconds':time.perf_counter()-tick,
        'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
        'sha256':{name+'.npy':sha(out/(name+'.npy')) for name in arrays}}
    write_json(out/'metadata.json',record)
    return record


def load_pack(root,study,name):
    cache=root/study['disk_inputs']; meta=json.loads((cache/'metadata.json').read_text())
    original=json.loads((root/study['parent_output']/'cache/development/metadata.json').read_text())
    fm=json.loads((root/study['output']/'features/metadata.json').read_text())
    split=json.loads((root/study['split']).read_text())
    assert meta['status']==fm['status']=='complete'
    assert meta['ids']==fm['ids']==sorted(split['fit_ids']+split['validation_ids'])
    assert meta['parent_cache_sha256']==fm['parent_cache_sha256']==original['cache_sha256']
    assert meta['split_sha256']==fm['split_sha256']==sha(root/study['split'])
    for filename,digest in meta['sha256'].items(): verify(cache/filename,digest)
    ids,scenes=meta['ids'],meta['scenes']; positions={index:n for n,index in enumerate(ids)}
    fits=[positions[i] for i in split['fit_ids']]; vals=[positions[i] for i in split['validation_ids']]
    assert len(fits)==577 and len(vals)==218 and not {scenes[i] for i in fits}&{scenes[i] for i in vals}
    pack={key:DiskArray(cache/(key+'.npy')) for key in ('original','truth','mask','band')}
    pack.update(ids=ids,scenes=scenes,fitting=fits,validation=vals,metadata=original,feature_metadata=fm)
    internal={}
    for alias,source in study['candidates'][name]['source_map'].items():
        path=root/study['output']/'features'/(source+'.npy'); verify(path,fm['sha256'][source+'.npy'])
        internal[alias]=DiskArray(path)
    return pack,DiskArray(cache/'inputs.npy'),internal
