import sys,unittest
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from p3_depth.residual.model import Refiner,corrected_depth
from p3_depth.residual_inputs.model import SingleStage,MultiStage
from p3_depth.residual_inputs.data import observable_inputs
from p3_depth.residual.deployment import image_inputs

class InputExtensionTests(unittest.TestCase):
    def test_single_stage_preserves_parent_parameters_and_forward(self):
        torch.manual_seed(7);parent=Refiner(4,'cnn',256)
        torch.manual_seed(7);model=SingleStage(4,'init',256)
        for name,value in parent.state_dict().items():self.assertTrue(torch.equal(value,model.state_dict()[name]))
        torch.nn.init.normal_(parent.output.weight,std=.001);model.load_state_dict(parent.state_dict())
        x=torch.randn(2,4,24,32);f=torch.randn(2,256,3,4)
        self.assertTrue(torch.equal(parent(x,f),model(x,{'init':f})))
    def test_multistage_zero_control_and_nonzero_branch_gradient(self):
        cfg={'branches':['init','late'],'widths':[8,8]};model=MultiStage(4,cfg,{'init':256,'late':64})
        x=torch.randn(2,4,24,32);v={'init':torch.randn(2,256,3,4),'late':torch.randn(2,64,12,16)}
        self.assertTrue(torch.equal(model(x,v),torch.zeros(2,1,24,32)))
        torch.nn.init.normal_(model.body.output.weight,std=.01)
        model(x,v).square().mean().backward();self.assertGreater(model.projections['late'].weight.grad.abs().sum().item(),0)
        model.zero=['late'];one=model(x,v);v['late']=torch.randn_like(v['late'])*100
        self.assertTrue(torch.equal(one,model(x,v)))
    def test_observable_inputs_match_previous_and_do_not_accept_gt(self):
        rng=np.random.default_rng(3);rgb=rng.integers(0,256,(48,64,3),dtype=np.uint8);d=rng.uniform(.1,10,(48,64)).astype(np.float32);f=np.ones((256,3,4),np.float16)
        cfg={'inputs':'frozen_features','base':'original','extra_inputs':[],'drop_rgb':False}
        a=observable_inputs(rgb,d,None,cfg,[24,32],None)
        b=image_inputs(rgb,d,cfg,[24,32],{},internal=f)[0][0]
        np.testing.assert_array_equal(a,b)
        cfg['drop_rgb']=True;np.testing.assert_array_equal(observable_inputs(rgb,d,None,cfg,[24,32],None),b[3:])

if __name__=='__main__':unittest.main()
