import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.normalization import LayerNorm


class TestLayerNorm(unittest.TestCase):
    def setUp(self):
        self.embedding_dim = 6
        self.ln = LayerNorm(self.embedding_dim)

    def test_output_shape_matches_input(self):
        x = torch.randn(3, 5, self.embedding_dim)
        out = self.ln(x)
        self.assertEqual(out.shape, x.shape)

    def test_default_init_produces_zero_mean_unit_variance(self):
        # gamma=1, beta=0 at init, so the raw normalization should be visible directly
        x = torch.randn(4, 7, self.embedding_dim) * 50 + 100  # arbitrary scale/offset
        out = self.ln(x)
        mean = out.mean(dim=-1)
        var = out.var(dim=-1, unbiased=False)
        self.assertTrue(torch.allclose(mean, torch.zeros_like(mean), atol=1e-5))
        self.assertTrue(torch.allclose(var, torch.ones_like(var), atol=1e-4))

    def test_matches_pytorch_reference_implementation(self):
        torch.manual_seed(0)
        x = torch.randn(2, 4, self.embedding_dim)
        ref = nn.LayerNorm(self.embedding_dim, eps=self.ln.eps)
        with torch.no_grad():
            ref.weight.copy_(self.ln.gamma)
            ref.bias.copy_(self.ln.beta)
        self.assertTrue(torch.allclose(self.ln(x), ref(x), atol=1e-6))

    def test_learned_gamma_and_beta_actually_apply(self):
        with torch.no_grad():
            self.ln.gamma.fill_(2.0)
            self.ln.beta.fill_(3.0)
        x = torch.randn(2, 3, self.embedding_dim)
        out = self.ln(x)
        mean = (out - 3.0) / 2.0  # undo the known scale/shift
        self.assertTrue(torch.allclose(mean.mean(dim=-1), torch.zeros(2, 3), atol=1e-5))

    def test_normalization_is_per_position_and_independent_of_other_tokens(self):
        # Unlike BatchNorm, changing one token must not affect another token's
        # normalized output, even within the same sequence/batch.
        torch.manual_seed(0)
        x = torch.randn(2, 4, self.embedding_dim)
        out_before = self.ln(x)

        x_modified = x.clone()
        x_modified[0, 1] = torch.randn(self.embedding_dim) * 1000
        out_after = self.ln(x_modified)

        unaffected = [(b, t) for b in range(2) for t in range(4) if (b, t) != (0, 1)]
        for b, t in unaffected:
            self.assertTrue(torch.allclose(out_before[b, t], out_after[b, t], atol=1e-5))

    def test_gradients_flow_to_gamma_beta_and_input(self):
        x = torch.randn(2, 3, self.embedding_dim, requires_grad=True)
        loss = self.ln(x).sum()
        loss.backward()
        self.assertIsNotNone(self.ln.gamma.grad)
        self.assertIsNotNone(self.ln.beta.grad)
        self.assertIsNotNone(x.grad)


if __name__ == "__main__":
    unittest.main()
