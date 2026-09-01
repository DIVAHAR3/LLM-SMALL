import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.residual import ResidualConnection


class ZeroSublayer(nn.Module):
    """Always outputs zero, derived from x only for its shape -- lets us
    isolate the pure identity/skip path."""

    def forward(self, x):
        return torch.zeros_like(x)


class ConstantSublayer(nn.Module):
    """Outputs a fixed value that does NOT depend on x's values, only its
    shape -- so d(sublayer(x))/dx is exactly zero everywhere. Any gradient
    the input receives must have come through the residual's "+x" path."""

    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return torch.zeros_like(x) + self.value


class DoubleSublayer(nn.Module):
    def forward(self, x):
        return x * 2


class TestResidualConnection(unittest.TestCase):
    def test_output_shape_matches_input(self):
        res = ResidualConnection(DoubleSublayer())
        x = torch.randn(3, 5, 8)
        out = res(x)
        self.assertEqual(out.shape, x.shape)

    def test_zero_sublayer_preserves_input_exactly(self):
        res = ResidualConnection(ZeroSublayer())
        x = torch.randn(2, 4, 6)
        out = res(x)
        self.assertTrue(torch.equal(out, x))

    def test_output_equals_input_plus_sublayer_output(self):
        res = ResidualConnection(DoubleSublayer())
        x = torch.randn(2, 4, 6)
        out = res(x)
        self.assertTrue(torch.allclose(out, x + 2 * x))

    def test_gradient_flows_through_identity_path_even_when_sublayer_gradient_is_zero(self):
        res = ResidualConnection(ConstantSublayer(value=5.0))
        x = torch.randn(2, 3, 4, requires_grad=True)
        out = res(x)
        out.sum().backward()
        # d(sum(x + constant))/dx == 1 everywhere: proves the skip path alone
        # carries gradient back to x, independent of the sublayer's own gradient.
        self.assertTrue(torch.allclose(x.grad, torch.ones_like(x)))


if __name__ == "__main__":
    unittest.main()
