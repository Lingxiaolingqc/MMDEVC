"""Reuse frozen parent predictions; build only the declared observable inputs."""
import json
import cv2,h5py,numpy as np
from p3_depth.residual.data import load_pack as parent_pack
from p3_depth.residual.records import sha,identity,write_json
from p3_depth.residual.job_inputs import edges_at_resolution
from p3_depth.data.nyuv2 import NYUv2Labeled

def load_pack(root,study):
    parent=json.loads((root/study['parent_config']).read_text(encoding='utf8'))
    pack=parent_pack(root,parent)
    # Inputs are assembled from the immutable source/cache below. These parent
    # arrays are never consumed by this extension and need not stay resident.
    for key in ('rgb','flip','confidence'):pack.pop(key)
    return pack

def fit_job_base(root,study,job,pack):
    return pack['original'],{'scale_used':False,'fitting_ids':job['fit_ids'],'cache_sha256':pack['metadata']['cache_sha256']}

def observable_inputs(rgb,depth,confidence,config,hw,edge_calibration):
    width,height=hw[::-1]
    image=cv2.resize(rgb,(width,height),interpolation=cv2.INTER_AREA).transpose(2,0,1).astype(np.float32)/127.5-1
    log=np.log(np.maximum(cv2.resize(depth,(width,height),interpolation=cv2.INTER_AREA),.001))[None]/2
    values=[log] if config['drop_rgb'] else [image,log]
    for name in config['extra_inputs']:
        if name=='confidence':
            conf=cv2.resize(confidence,(width,height),interpolation=cv2.INTER_AREA)
            values.append((np.clip(np.log(np.maximum(conf,1e-6)),-5,5)/5)[None])
        else:
            edge=edges_at_resolution(rgb,depth,edge_calibration,hw)
            for key in (['rgb_sobel','depth_sobel'] if name=='joint_edges' else [name]):values.append(edge[key][None])
    result=np.concatenate(values).astype(np.float32)
    assert np.isfinite(result).all();return result

def prepare_job_features(root,study,job,pack,base,calibration):
    cfg=job['candidate'];parent=root/study['parent_output']
    edge_meta=json.loads((parent/'edge_features/metadata.json').read_text(encoding='utf8'))
    assert edge_meta['config']['cache_sha256']==pack['metadata']['cache_sha256']
    spec={'ids':pack['ids'],'drop_rgb':cfg['drop_rgb'],'extras':cfg['extra_inputs'],'hw':job['feature_hw'],
          'parent_cache':pack['metadata']['cache_sha256'],'edge_calibration':edge_meta['calibration']}
    output=root/study['output']/'derived_inputs'/identity(spec);output.mkdir(parents=True,exist_ok=True)
    path=output/'features.npy';meta_path=output/'metadata.json'
    if meta_path.exists():
        meta=json.loads(meta_path.read_text(encoding='utf8'));assert meta['spec']==spec and sha(path)==meta['sha256']
    else:
        array=None
        with NYUv2Labeled(root/study['data']) as source,h5py.File(pack['cache_path'],'r') as cache:
            for i,index in enumerate(pack['ids']):
                rgb=source.file['images'][index].transpose(2,1,0).copy()
                value=observable_inputs(rgb,base[i],cache[str(index)]['confidence'][:],cfg,job['feature_hw'],edge_meta['calibration'])
                if array is None:array=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(base),*value.shape))
                array[i]=value
            array.flush();del array
        meta={'spec':spec,'sha256':sha(path)};write_json(meta_path,meta)
    stages=root/study['output']/'stage_cache/development';stage_meta=json.loads((stages/'metadata.json').read_text(encoding='utf8'))
    assert stage_meta['status']=='complete' and stage_meta['spec']['ids']==pack['ids']
    internal={name:np.load(stages/(name+'.npy'),mmap_mode='r') for name in cfg['branches']}
    # One cached verification per process avoids hashing13GB for each candidate.
    for name in internal:
        key=(str(stages),name,stage_meta['sha256'][name+'.npy'])
        if key not in VERIFIED:
            assert sha(stages/(name+'.npy'))==key[-1];VERIFIED.add(key)
    return np.load(path,mmap_mode='r'),internal,{'inputs':meta,'stages':stage_meta}

VERIFIED=set()
