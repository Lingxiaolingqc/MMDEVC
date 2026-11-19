import unittest
import torch
from p3_depth.adaptation_study.lora_numerics import FiniteAcos


class RayAngleGradientTests(unittest.TestCase):
    def test_forward_and_pole_gradients(self):
        for dtype in (torch.float16, torch.float32, torch.float64):
            x = torch.tensor([-1., -.8, 0., .8, 1.], dtype=dtype, requires_grad=True)
            y = FiniteAcos.apply(x)
            self.assertTrue(torch.equal(y, torch.acos(x)))
            y.sum().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertTrue((x.grad < 0).all())

    def test_interior_derivative(self):
        x = torch.tensor([-.9, -.2, .3, .9], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(FiniteAcos.apply, (x,)))


if __name__ == '__main__':
    unittest.main()
