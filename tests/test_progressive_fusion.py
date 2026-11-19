import unittest
import torch
from p3_depth.adaptation_study.model import make_model
from p3_depth.residual_inputs.model import make_model as old_model


class ProgressiveFusionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(91)
        self.inputs = torch.randn(2,4,32,40)
        self.features = {'init':torch.randn(2,256,4,5), 'mid':torch.randn(2,128,8,10),
                         'late':torch.randn(2,64,16,20)}

    def config(self, mode, three=False):
        return {'fusion':mode, 'branches':['init','mid','late'] if three else ['init','late'],
                'widths':[6,5,5] if three else [8,8]}

    def test_flat_preserves_existing_model(self):
        cfg = self.config('flat')
        old = old_model(4,cfg,self.features)
        new = make_model(4,cfg,self.features)
        new.load_state_dict(old.state_dict())
        with torch.no_grad():
            old.body.output.weight.normal_(0,.1)
        new.load_state_dict(old.state_dict())
        self.assertTrue(torch.equal(old(self.inputs,self.features),new(self.inputs,self.features)))

    def test_common_grid_has_identical_parameters(self):
        for three in (False,True):
            progressive = make_model(4,self.config('progressive',three),self.features)
            common = make_model(4,self.config('common_grid',three),self.features)
            common.load_state_dict(progressive.state_dict())
            self.assertEqual(sum(p.numel() for p in progressive.parameters()),
                             sum(p.numel() for p in common.parameters()))

    def test_zero_initialization_and_all_branch_gradients(self):
        for mode in ('flat','progressive','gated','common_grid'):
            for three in (False,True):
                cfg = self.config(mode,three)
                model = make_model(4,cfg,self.features)
                self.assertEqual(model(self.inputs,self.features).abs().max().item(),0)
                with torch.no_grad():
                    model.body.output.weight.normal_(0,.1)
                value = model(self.inputs,self.features)
                self.assertEqual(value.shape,(2,1,32,40))
                value.square().mean().backward()
                for name in cfg['branches']:
                    grad = model.projections[name].weight.grad
                    self.assertTrue(torch.isfinite(grad).all())
                    self.assertGreater(grad.abs().sum().item(),0)
                if mode == 'gated':
                    self.assertTrue(all(g.weight.grad.abs().sum().item()>0 for g in model.gates))


if __name__ == '__main__':
    unittest.main()
