"""Bounded-resident read-only arrays and native NYUv2 RGB, without full RAM caches."""
import json
import numpy as np
import h5py
from .protocol import load

class DiskArray:
    def __init__(self,path):
        self.path=path
        a=np.load(path,mmap_mode='r');self.shape,self.dtype=a.shape,a.dtype;a._mmap.close()
    def __getitem__(self,index):
        a=np.load(self.path,mmap_mode='r')
        try:
            if isinstance(index,(list,np.ndarray)):
                out=np.empty((len(index),*self.shape[1:]),dtype=self.dtype)
                for j,i in enumerate(index):out[j]=a[int(i)]
                return out
            return np.array(a[index],copy=True)
        finally:a._mmap.close()

class Pack(dict):
    def __init__(self,root):
        cfg=load(root);disk=root/cfg['disk_inputs'];stage=root/cfg['features']
        meta=json.loads((disk/'metadata.json').read_text())
        super().__init__({k:DiskArray(disk/(k+'.npy')) for k in ['original','truth','mask','band','inputs']})
        self.update(ids=meta['ids'],scenes=meta['scenes'])
        self.stages={k:DiskArray(stage/(k+'.npy')) for k in ['init','mid','late']}
        self.pos={i:n for n,i in enumerate(self['ids'])}
        self.file=h5py.File(root/cfg['data'],'r')
    def positions(self,ids):return [self.pos[i] for i in ids]
    def tensor(self,key,positions):
        import torch
        return torch.from_numpy(self[key][positions]).cuda()
    def rgb(self,positions):
        import torch
        values=np.stack([self.file['images'][self['ids'][i]].transpose(0,2,1) for i in positions])
        return torch.from_numpy(values).cuda()
    def cached(self,positions):
        import torch
        return (self.tensor('inputs',positions),
                {k:torch.from_numpy(a[positions]).cuda().float() for k,a in self.stages.items()},
                self.tensor('original',positions)[:,None])
    def close(self):self.file.close()
