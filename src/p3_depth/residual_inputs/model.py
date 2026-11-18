"""Fixed CNN body and a total16-channel native-stage projection budget."""
import torch
from torch import nn
from torch.nn import functional as F
from p3_depth.residual.model import Refiner

class SingleStage(Refiner):
    def __init__(self,inputs,name,channels,zero=False):
        super().__init__(inputs,'cnn',channels);self.branch=name;self.zero=zero
    def forward(self,inputs,internal):
        value=internal[self.branch]
        return super().forward(inputs,torch.zeros_like(value) if self.zero else value)

class MultiStage(nn.Module):
    def __init__(self,inputs,config,channels):
        super().__init__();self.branches=config['branches'];self.zero=config.get('zero_branches',[])
        assert sum(config['widths'])==16
        self.projections=nn.ModuleDict({name:nn.Conv2d(channels[name],width,1) for name,width in zip(self.branches,config['widths'])})
        self.body=Refiner(inputs+16,'cnn',0)
    def forward(self,inputs,internal):
        values=[inputs]
        for name in self.branches:
            value=internal[name]
            if name in self.zero:value=torch.zeros_like(value)
            values.append(F.interpolate(self.projections[name](value),inputs.shape[-2:],mode='bilinear',align_corners=False))
        return self.body(torch.cat(values,1))

def make_model(inputs,config,internal):
    channels={name:array.shape[1] for name,array in internal.items()}
    if len(config['branches'])==1:
        name=config['branches'][0];return SingleStage(inputs,name,channels[name],name in config.get('zero_branches',[]))
    return MultiStage(inputs,config,channels)
