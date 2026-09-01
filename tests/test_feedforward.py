import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.feedforward import FeedForward


class TestFeedForward(unittest.TestCase):
    def setUp(self):
        self.embedding_dim = 8
        self.hidden_dim = 32
        self.ffn = FeedForward(self.embedding_dim, self.hidden_dim, dropout=0.0)
        self.ffn.eval()

    def test_output_shape_matches_input_embedding_dim(self):
        x = torch.randn(4, 10, self.embedding_dim)
        out = self.ffn(x)
        self.assertEqual(out.shape, x.shape)

    def test_hidden_dim_is_actually_used(self):
        self.assertEqual(self.ffn.fc1.out_features, self.hidden_dim)
        self.assertEqual(self.ffn.fc2.in_features, self.hidden_dim)

    def test_configurable_dims_are_independent_of_hidden_dim(self):
        ffn = FeedForward(embedding_dim=6, hidden_dim=100, dropout=0.0)
        x = torch.randn(2, 3, 6)
        out = ffn(x)
        self.assertEqual(out.shape, (2, 3, 6))

    def test_position_wise_independence(self):
        # The defining property vs. attention: each position is transformed
        # in isolation, so the same vector at a given position produces the
        # same output regardless of what surrounds it in the sequence.
        torch.manual_seed(0)
        shared_vector = torch.randn(self.embedding_dim)

        seq_a = torch.randn(1, 4, self.embedding_dim)
        seq_a[0, 1] = shared_vector
        seq_b = torch.randn(1, 4, self.embedding_dim)
        seq_b[0, 2] = shared_vector

        out_a = self.ffn(seq_a)
        out_b = self.ffn(seq_b)
        self.assertTrue(torch.allclose(out_a[0, 1], out_b[0, 2], atol=1e-6))

    def test_gradients_flow_to_both_linear_layers(self):
        ffn = FeedForward(self.embedding_dim, self.hidden_dim, dropout=0.0)
        x = torch.randn(2, 5, self.embedding_dim)
        loss = ffn(x).sum()
        loss.backward()
        for name, param in ffn.named_parameters():
            self.assertIsNotNone(param.grad, f"{name} got no gradient")

    def test_rejects_unsupported_activation(self):
        with self.assertRaises(ValueError):
            FeedForward(self.embedding_dim, self.hidden_dim, activation="sigmoid")

    def test_relu_activation_selectable(self):
        ffn = FeedForward(self.embedding_dim, self.hidden_dim, activation="relu")
        self.assertIsInstance(ffn.activation, torch.nn.ReLU)

    def test_dropout_module_configured(self):
        ffn = FeedForward(self.embedding_dim, self.hidden_dim, dropout=0.4)
        self.assertIsInstance(ffn.dropout, torch.nn.Dropout)
        self.assertEqual(ffn.dropout.p, 0.4)


if __name__ == "__main__":
    unittest.main()
