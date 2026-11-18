"""Memory-bounded, reproducible features for combined/resolution/data-size controls."""
import json
from contextlib import ExitStack
import cv2
import h5py
import numpy as np
from p3_depth.data.nyuv2 import NYUv2Labeled
from .data import base_for, fit_scale, features_for
from .edges import derivatives, depth_canny
from .records import sha, identity, write_json


def fit_job_base(root, study, job, pack):
    positions={index:i for i,index in enumerate(pack['ids'])}
    fitting=[positions[index] for index in job['fit_ids']]
    name=job['candidate']['base']
    calibration={'fitting_ids':job['fit_ids'],'cache_sha256':pack['metadata']['cache_sha256'],
                 'scale_used':name in ('scale','flip_scale')}
    if name in ('scale','flip_scale'):
        key=identity({'fitting_ids':job['fit_ids'],'base':name,'cache':pack['metadata']['cache_sha256']})
        path=root/study['output']/'job_calibration'/(key+'.json')
        if path.exists():
            calibration=json.loads(path.read_text(encoding='utf-8'))
        else:
            raw=pack['original'] if name=='scale' else .75*pack['original']+.25*pack['flip']
            calibration[name]=fit_scale(raw,pack['truth'],pack['mask'],fitting)
            write_json(path,calibration)
    return base_for(pack,name,calibration),calibration


def edges_at_resolution(rgb, original, calibration, hw):
    gray,rm,dm,grads=derivatives(rgb,original)
    values={'rgb_sobel':np.clip(rm/max(calibration['rgb_magnitude_normalizer'],1e-8),0,5),
            'depth_sobel':np.clip(dm/max(calibration['depth_magnitude_normalizer'],1e-8),0,5),
            'rgb_canny':(cv2.Canny(gray,.4*calibration['rgb_high'],calibration['rgb_high'],L2gradient=True)>0).astype(np.float32),
            'depth_canny':depth_canny(grads,calibration['depth_high']).astype(np.float32)}
    # Match the screening HDF5 representation (float16 edge maps, then float32
    # network inputs), including at higher resolution.
    return {key:cv2.resize(value,tuple(hw[::-1]),interpolation=cv2.INTER_AREA).astype(np.float16).astype(np.float32) for key,value in values.items()}


def assemble_input_chunk(pack, config, base, hw, edge=None):
    """Inference-only fields; no truth/mask/boundary labels accepted or read."""
    values,internal=features_for(pack,config,base,hw,edge)
    if config.get('drop_rgb',False):
        if config['inputs']=='depth_only':
            raise ValueError('RGB already absent')
        values=values[:,3:]
    extras=config.get('extra_inputs',[])
    if len(extras)!=len(set(extras)):
        raise ValueError('Duplicate additional input groups')
    for name in extras:
        if name not in ('confidence','coordinates','flip','flip_disagreement','rgb_sobel','depth_sobel','rgb_canny','depth_canny','joint_edges'):
            raise ValueError('Unknown extra input')
        if name==config['inputs']:
            raise ValueError('Input group added twice')
        extra,_=features_for(pack,{'inputs':name},base,hw,edge)
        values=np.concatenate([values,extra[:,4:]],axis=1)
    if config.get('internal_mode','true')=='zeros':
        if internal is None:
            raise ValueError('Cannot disable absent internal features')
        internal=np.zeros_like(internal)
    return values,internal


def prepare_job_features(root, study, job, pack, base, calibration):
    """Store float32 input maps in shared memmaps rather than 4GB heap allocations."""
    cfg=job['candidate'];hw=job['feature_hw'];height,width=hw
    uses_edges=cfg['inputs'] in ('rgb_sobel','depth_sobel','rgb_canny','depth_canny','joint_edges') or any('sobel' in x or 'canny' in x or x=='joint_edges' for x in cfg.get('extra_inputs',[]))
    edge_meta=None
    if uses_edges:
        edge_meta=json.loads((root/study['output']/'edge_features/metadata.json').read_text(encoding='utf-8'))
        if edge_meta['config']['cache_sha256']!=pack['metadata']['cache_sha256']:
            raise ValueError('Edge calibration differs from prediction source')
        # Edge calibration uses only unlabeled fitting RGB/predictions. For data-size
        # experiments involving edges, require an explicit subset recalibration first.
        if job['phase']=='data_size' and job['fit_ids']!=edge_meta['config']['fitting_ids']:
            raise ValueError('Subset edge calibration not supplied; do not silently use extra fitting images')
    spec={'ids':pack['ids'],'cache_sha256':pack['metadata']['cache_sha256'],
          'inputs':cfg['inputs'],'base':cfg['base'],'scale':calibration.get(cfg['base'],{}).get('scale',1.),
          'drop_rgb':cfg.get('drop_rgb',False),'extra_inputs':cfg.get('extra_inputs',[]),
          'internal_mode':cfg.get('internal_mode','true'),'feature_hw':hw,'dtype':'float32',
          'edge_calibration':edge_meta['calibration'] if edge_meta else None,'schema':1}
    directory=root/study['output']/'derived_inputs'/identity(spec)
    directory.mkdir(parents=True,exist_ok=True)
    meta_path=directory/'metadata.json'
    if meta_path.exists():
        meta=json.loads(meta_path.read_text(encoding='utf-8'))
        if meta['spec']!=spec or any(sha(directory/name)!=value for name,value in meta['sha256'].items()):
            raise ValueError('Derived input cache mismatch')
        values=np.load(directory/'features.npy',mmap_mode='r',allow_pickle=False)
        latent=np.load(directory/'internal.npy',mmap_mode='r',allow_pickle=False) if (directory/'internal.npy').exists() else None
        return values,latent,meta
    with ExitStack() as stack:
        data=stack.enter_context(NYUv2Labeled(root/study['data']))
        original_cache=stack.enter_context(h5py.File(pack['cache_path'],'r'))
        values_out=internal_out=None
        for start in range(0,len(pack['ids']),8):
            stop=min(start+8,len(pack['ids']));idxs=list(range(start,stop))
            rgb=[];confidence=[];edge={key:[] for key in ('rgb_sobel','rgb_canny','depth_sobel','depth_canny')} if uses_edges else None
            for i in idxs:
                index=pack['ids'][i]
                image=data.file['images'][index].transpose(2,1,0).copy()
                rgb.append(cv2.resize(image,(width,height),interpolation=cv2.INTER_AREA).transpose(2,0,1).astype(np.float32)/127.5-1)
                conf=cv2.resize(original_cache[str(index)]['confidence'][:],(width,height),interpolation=cv2.INTER_AREA)
                confidence.append(np.clip(np.log(np.maximum(conf,1e-6)),-5,5)/5)
                if uses_edges:
                    prepared=edges_at_resolution(image,pack['original'][i],edge_meta['calibration'],hw)
                    for key,value in prepared.items():edge[key].append(value)
            mini={'ids':pack['ids'][start:stop],'cache_path':pack['cache_path'],
                  'original':pack['original'][start:stop],'flip':pack['flip'][start:stop],
                  'rgb':np.stack(rgb),'confidence':np.stack(confidence)}
            chunk,latent=assemble_input_chunk(mini,cfg,base[start:stop],hw,
                    {key:np.stack(value) for key,value in edge.items()} if edge else None)
            if values_out is None:
                values_out=np.lib.format.open_memmap(directory/'features.npy',mode='w+',dtype=np.float32,shape=(len(base),*chunk.shape[1:]))
                if latent is not None:
                    internal_out=np.lib.format.open_memmap(directory/'internal.npy',mode='w+',dtype=np.float16,shape=(len(base),*latent.shape[1:]))
            values_out[start:stop]=chunk
            if internal_out is not None:internal_out[start:stop]=latent
            if stop%80==0 or stop==len(base):print(f'INPUTS {job["run_id"]} {stop}/{len(base)} {hw}',flush=True)
        values_out.flush()
        if internal_out is not None:internal_out.flush()
        del values_out,internal_out
    names=['features.npy']+(['internal.npy'] if (directory/'internal.npy').exists() else [])
    meta={'status':'complete','spec':spec,'sha256':{name:sha(directory/name) for name in names},
          'bytes':{name:(directory/name).stat().st_size for name in names}}
    write_json(meta_path,meta)
    return np.load(directory/'features.npy',mmap_mode='r',allow_pickle=False),\
        (np.load(directory/'internal.npy',mmap_mode='r',allow_pickle=False) if 'internal.npy' in names else None),meta
