import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.attention import MultiHeadAttention, SingleHeadAttention, scaled_dot_product_attention


class TestScaledDotProductAttention(unittest.TestCase):
    def test_attention_weights_sum_to_one(self):
        q = torch.randn(2, 5, 8)
        k = torch.randn(2, 5, 8)
        v = torch.randn(2, 5, 8)
        _, weights = scaled_dot_product_attention(q, k, v)
        sums = weights.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-6))

    def test_non_causal_matches_hand_computed_values(self):
        # T=2, d_k=1: both queries/keys equal -> uniform 0.5/0.5 attention -> output is the mean of v
        q = torch.tensor([[[1.0], [1.0]]])
        k = torch.tensor([[[1.0], [1.0]]])
        v = torch.tensor([[[10.0], [20.0]]])
        out, weights = scaled_dot_product_attention(q, k, v, causal=False)
        self.assertTrue(torch.allclose(weights, torch.full((1, 2, 2), 0.5)))
        self.assertTrue(torch.allclose(out, torch.tensor([[[15.0], [15.0]]])))

    def test_causal_matches_hand_computed_values(self):
        # same q/k/v as above, but causal: position 0 can only see itself -> output[0] == v[0]
        q = torch.tensor([[[1.0], [1.0]]])
        k = torch.tensor([[[1.0], [1.0]]])
        v = torch.tensor([[[10.0], [20.0]]])
        out, weights = scaled_dot_product_attention(q, k, v, causal=True)
        self.assertTrue(torch.allclose(weights[0, 0], torch.tensor([1.0, 0.0])))
        self.assertTrue(torch.allclose(weights[0, 1], torch.tensor([0.5, 0.5])))
        self.assertTrue(torch.allclose(out, torch.tensor([[[10.0], [15.0]]])))

    def test_causal_mask_zeroes_future_attention_weights(self):
        q = torch.randn(1, 4, 8)
        k = torch.randn(1, 4, 8)
        v = torch.randn(1, 4, 8)
        _, weights = scaled_dot_product_attention(q, k, v, causal=True)
        future_mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
        self.assertTrue(torch.all(weights[0][future_mask] == 0))


class TestCausalityIsFunctional(unittest.TestCase):
    """The strongest test of masking: changing a FUTURE token must not change
    an EARLIER position's output at all, not just have near-zero weight."""

    def test_changing_future_token_does_not_affect_past_output(self):
        torch.manual_seed(0)
        head = SingleHeadAttention(embedding_dim=6, head_dim=4, causal=True)
        head.eval()
        x = torch.randn(1, 5, 6)
        out_before = head(x)

        x_modified = x.clone()
        x_modified[0, -1, :] = torch.randn(6) * 100  # drastically change only the last (future) token
        out_after = head(x_modified)

        # every position except the last one must be completely unaffected
        self.assertTrue(torch.allclose(out_before[0, :-1], out_after[0, :-1], atol=1e-6))
        # the last position, which now attends to a changed token, should differ
        self.assertFalse(torch.allclose(out_before[0, -1], out_after[0, -1]))

    def test_non_causal_future_token_does_affect_past_output(self):
        # sanity contrast: without masking, changing any token can affect any output
        torch.manual_seed(0)
        head = SingleHeadAttention(embedding_dim=6, head_dim=4, causal=False)
        head.eval()
        x = torch.randn(1, 5, 6)
        out_before = head(x)

        x_modified = x.clone()
        x_modified[0, -1, :] = torch.randn(6) * 100
        out_after = head(x_modified)

        self.assertFalse(torch.allclose(out_before[0, :-1], out_after[0, :-1]))


class TestSingleHeadAttention(unittest.TestCase):
    def test_output_shape(self):
        head = SingleHeadAttention(embedding_dim=8, head_dim=5)
        x = torch.randn(3, 10, 8)
        out = head(x)
        self.assertEqual(out.shape, (3, 10, 5))

    def test_gradients_flow_to_projections(self):
        head = SingleHeadAttention(embedding_dim=8, head_dim=5)
        x = torch.randn(2, 4, 8)
        loss = head(x).sum()
        loss.backward()
        for name, param in head.named_parameters():
            self.assertIsNotNone(param.grad, f"{name} got no gradient")


class TestMultiHeadAttention(unittest.TestCase):
    def test_output_shape_matches_input_embedding_dim(self):
        mha = MultiHeadAttention(embedding_dim=16, num_heads=4)
        x = torch.randn(2, 7, 16)
        out = mha(x)
        self.assertEqual(out.shape, (2, 7, 16))

    def test_rejects_embedding_dim_not_divisible_by_num_heads(self):
        with self.assertRaises(ValueError):
            MultiHeadAttention(embedding_dim=10, num_heads=3)

    def test_causal_masking_blocks_future_tokens(self):
        torch.manual_seed(0)
        mha = MultiHeadAttention(embedding_dim=16, num_heads=4, causal=True, dropout=0.0)
        mha.eval()
        x = torch.randn(1, 5, 16)
        out_before = mha(x)

        x_modified = x.clone()
        x_modified[0, -1, :] = torch.randn(16) * 100
        out_after = mha(x_modified)

        self.assertTrue(torch.allclose(out_before[0, :-1], out_after[0, :-1], atol=1e-6))

    def test_gradients_flow_through_all_heads(self):
        mha = MultiHeadAttention(embedding_dim=16, num_heads=4, dropout=0.0)
        x = torch.randn(2, 6, 16)
        loss = mha(x).sum()
        loss.backward()
        for name, param in mha.named_parameters():
            self.assertIsNotNone(param.grad, f"{name} got no gradient")

    def test_dropout_module_configured(self):
        mha = MultiHeadAttention(embedding_dim=16, num_heads=4, dropout=0.2)
        self.assertIsInstance(mha.dropout, torch.nn.Dropout)
        self.assertEqual(mha.dropout.p, 0.2)


if __name__ == "__main__":
    unittest.main()
