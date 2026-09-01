import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.transformer import TransformerBlock


class TestTransformerBlock(unittest.TestCase):
    def setUp(self):
        self.embedding_dim = 16
        self.num_heads = 4
        self.ffn_hidden_dim = 64
        self.block = TransformerBlock(self.embedding_dim, self.num_heads, self.ffn_hidden_dim, dropout=0.0)
        self.block.eval()

    def test_forward_pass_runs_and_shape_is_preserved(self):
        x = torch.randn(3, 10, self.embedding_dim)
        out = self.block(x)
        self.assertEqual(out.shape, x.shape)

    def test_causality_is_preserved_through_the_whole_block(self):
        # Same style of test as Phase 6's attention causality test, but now
        # exercising the composed block (LN + attn + residual + LN + FFN +
        # residual) to confirm nothing downstream leaks future information.
        torch.manual_seed(0)
        x = torch.randn(1, 6, self.embedding_dim)
        out_before = self.block(x)

        x_modified = x.clone()
        x_modified[0, -1, :] = torch.randn(self.embedding_dim) * 100
        out_after = self.block(x_modified)

        self.assertTrue(torch.allclose(out_before[0, :-1], out_after[0, :-1], atol=1e-5))
        self.assertFalse(torch.allclose(out_before[0, -1], out_after[0, -1]))

    def test_gradients_flow_to_every_parameter(self):
        block = TransformerBlock(self.embedding_dim, self.num_heads, self.ffn_hidden_dim, dropout=0.0)
        x = torch.randn(2, 5, self.embedding_dim)
        loss = block(x).sum()
        loss.backward()
        for name, param in block.named_parameters():
            self.assertIsNotNone(param.grad, f"{name} got no gradient")

    def test_stack_of_blocks_composes_cleanly(self):
        # Foreshadows Phase 10: N blocks stacked sequentially should still
        # run end to end and preserve shape.
        stack = nn.Sequential(
            *[TransformerBlock(self.embedding_dim, self.num_heads, self.ffn_hidden_dim, dropout=0.0) for _ in range(3)]
        )
        x = torch.randn(2, 8, self.embedding_dim)
        out = stack(x)
        self.assertEqual(out.shape, x.shape)

    def test_qkv_bias_flag_is_forwarded(self):
        block = TransformerBlock(self.embedding_dim, self.num_heads, self.ffn_hidden_dim, qkv_bias=True)
        attn = block.attn_block.sublayer[1]
        self.assertIsNotNone(attn.W_q.bias)


if __name__ == "__main__":
    unittest.main()
