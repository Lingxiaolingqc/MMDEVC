import unittest
import torch
from torch import nn
from p3_depth.adaptation_study.lora import LoRALinear,adapter_state,load_adapters,assert_adapters_only


class LoRAAdapterTests(unittest.TestCase):
    def test_zero_output_frozen_base_and_adapter_roundtrip(self):
        torch.manual_seed(7)
        base=nn.Linear(5,4)
        original={k:v.clone() for k,v in base.state_dict().items()}
        model=nn.Module();model.block=LoRALinear(base,2)
        x=torch.randn(3,5)
        self.assertTrue(torch.equal(base(x),model.block(x)))
        params=[p for p in model.parameters() if p.requires_grad]
        self.assertEqual(len(assert_adapters_only(model)),2)
        optimizer=torch.optim.SGD(params,lr=.1)
        for _ in range(3):
            optimizer.zero_grad();model.block(x).square().mean().backward();optimizer.step()
        for key,value in base.state_dict().items():self.assertTrue(torch.equal(value,original[key]))
        self.assertGreater(model.block.lora_A.weight.grad.abs().sum().item(),0)
        saved=adapter_state(model)
        fresh=nn.Module();fresh.block=LoRALinear(nn.Linear(5,4),2)
        fresh.block.base.load_state_dict(original)
        load_adapters(fresh,saved)
        self.assertTrue(torch.equal(model.block(x),fresh.block(x)))


if __name__=='__main__':unittest.main()
