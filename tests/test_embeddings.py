import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.embeddings import GPTEmbedding


class TestGPTEmbedding(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 10
        self.context_length = 6
        self.embedding_dim = 4
        self.emb = GPTEmbedding(self.vocab_size, self.context_length, self.embedding_dim, dropout=0.0)
        self.emb.eval()

    def test_output_shape(self):
        token_ids = torch.randint(0, self.vocab_size, (3, 5))
        out = self.emb(token_ids)
        self.assertEqual(out.shape, (3, 5, self.embedding_dim))

    def test_output_equals_token_plus_positional_embedding(self):
        token_ids = torch.tensor([[1, 2, 3]])
        out = self.emb(token_ids)
        expected = self.emb.token_embedding(token_ids) + self.emb.position_embedding(torch.arange(3))
        self.assertTrue(torch.allclose(out, expected))

    def test_same_token_different_position_gives_different_output(self):
        token_ids = torch.tensor([[4, 4, 4]])
        out = self.emb(token_ids)
        self.assertFalse(torch.allclose(out[0, 0], out[0, 1]))
        self.assertFalse(torch.allclose(out[0, 1], out[0, 2]))

    def test_identical_sequences_give_identical_output(self):
        token_ids = torch.tensor([[0, 0, 0], [0, 0, 0]])
        out = self.emb(token_ids)
        self.assertTrue(torch.allclose(out[0], out[1]))

    def test_raises_when_sequence_exceeds_context_length(self):
        token_ids = torch.randint(0, self.vocab_size, (1, self.context_length + 1))
        with self.assertRaises(ValueError):
            self.emb(token_ids)

    def test_dropout_module_configured(self):
        emb = GPTEmbedding(self.vocab_size, self.context_length, self.embedding_dim, dropout=0.3)
        self.assertIsInstance(emb.dropout, torch.nn.Dropout)
        self.assertEqual(emb.dropout.p, 0.3)


if __name__ == "__main__":
    unittest.main()
