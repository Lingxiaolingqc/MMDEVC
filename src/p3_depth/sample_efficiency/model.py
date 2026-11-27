"""F3 and encoder LoRA with explicit live autograd paths for joint training."""
import inspect
import torch
import yaml
from torch.nn import functional as F
from p3_depth.models.runtime import ImageModel
from p3_depth.adaptation_study.lora import inject, load_adapters, LoRALinear
from p3_depth.adaptation_study.model import make_model
from p3_depth.residual.model import corrected_depth

class Pipeline:
    def __init__(self,root,cfg,method,seed,stage_state=None):
        self.method=method;self.foundation=None;self.refiner=None;self.values={};self.handles=[]
        self.seed=seed;self.cfg=cfg;self.model_metadata=None;self.trace=False;self.last_paths={}
        if method in ('L','J','S','B0'):
            settings=yaml.safe_load((root/'configs/paths.local.yaml').read_text())
            adapter=ImageModel('unidepth',root,settings,'cuda',precision='fp16',threads=4)
            self.foundation=adapter.model;self.model_metadata=adapter.metadata
            if method!='B0':
                torch.manual_seed(seed)
                self.targets=inject(self.foundation,cfg['lora']['target'],cfg['lora']['rank'])
                for module in self.foundation.modules():
                    if isinstance(module,LoRALinear):module.eval()
                if method=='S':
                    assert stage_state is not None
                    load_adapters(self.foundation,stage_state['adapters'])
                    self.foundation.requires_grad_(False)
            else:self.foundation.requires_grad_(False)
            if method in ('J','S'):
                layer=self.foundation.pixel_decoder.depth_layer
                for key,index in [('mid',-2),('late',-1)]:
                    self.handles.append(layer.ups[index].register_forward_hook(self._hook(key)))
        if method in ('R','J','S'):
            # Shared F3 initialization, unaffected by foundation/adapter construction.
            torch.manual_seed(seed+17)
            class Shape:
                def __init__(self,c):self.shape=(0,c,0,0)
            channels={k:Shape(c) for k,c in [('init',256),('mid',128),('late',64)]}
            self.refiner=make_model(4,cfg['refiner'],channels).cuda()
        self.parameters=([p for p in self.foundation.parameters() if p.requires_grad] if self.foundation else [])
        if self.refiner:self.parameters+=list(self.refiner.parameters())
        self.assert_trainable()
    def assert_trainable(self):
        if self.foundation:
            actual={n for n,p in self.foundation.named_parameters() if p.requires_grad}
            expected={n for n,p in self.foundation.named_parameters() if '.lora_A.' in n or '.lora_B.' in n} if self.method in ('L','J') else set()
            assert actual==expected,('Unexpected foundation gradients',actual-expected,expected-actual)
    def _hook(self,key):
        def receive(module,args,output):self.values[key]=output
        return receive
    def infer(self,rgb):
        # inject() already established eval mode then froze original parameters.
        # Calling eval() again invokes the official ViT train(False) hook, which
        # re-enables norm gradients under frozen_stages=8.
        self.assert_trainable()
        assert not any(m.training for m in self.foundation.modules())
        body=inspect.unwrap(type(self.foundation).infer)
        with torch.autocast('cuda',dtype=torch.float16):return body(self.foundation,rgb)
    def predict(self,pack,positions):
        if self.method=='R':
            inputs,latent,depth=pack.cached(positions)
        else:
            self.values.clear()
            rgb=pack.rgb(positions)
            if self.method in ('S','B0'):
                with torch.no_grad():result=self.infer(rgb)
            else:result=self.infer(rgb)
            depth=result['depth'].float()
            if self.method in ('L','B0'):return depth
            latent={'init':result['depth_features'].float(),
                    **{k:v.float() for k,v in self.values.items()}}
            self.values.clear()
            image=pack.tensor('inputs',positions)[:,:3]
            log=F.interpolate(depth,size=self.cfg['feature_hw'],mode='area').clamp_min(.001).log()/2
            inputs=torch.cat([image,log],1)
        with torch.autocast('cuda',dtype=torch.float16):
            residual=self.refiner(inputs,latent)
        if self.trace:self.last_paths={'inputs':inputs,'latent':latent,'depth':depth}
        return corrected_depth(depth,residual,'additive')
    def state(self):
        return {'adapters':{n:p.detach().cpu().clone() for n,p in self.foundation.named_parameters()
                    if '.lora_A.' in n or '.lora_B.' in n} if self.foundation else {},
                'refiner':{n:t.detach().cpu().clone() for n,t in self.refiner.state_dict().items()} if self.refiner else {}}
    def restore(self,state):
        if self.foundation:
            params=dict(self.foundation.named_parameters())
            expected={n for n in params if '.lora_A.' in n or '.lora_B.' in n}
            assert expected==set(state['adapters'])
            with torch.no_grad():
                for n,v in state['adapters'].items():params[n].copy_(v)
        if self.refiner:self.refiner.load_state_dict(state['refiner'])
    def buffers(self):
        return {prefix+n:b.detach().cpu().clone()
                for prefix,module in [('foundation.',self.foundation),('refiner.',self.refiner)] if module
                for n,b in module.named_buffers()}
    def named_trainable(self):
        return [(prefix+n,p) for prefix,module in [('foundation.',self.foundation),('refiner.',self.refiner)] if module
                for n,p in module.named_parameters() if p.requires_grad]
    def close(self):
        for h in self.handles:h.remove()
        self.values.clear()
        self.last_paths.clear()
