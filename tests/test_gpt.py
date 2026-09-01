import json
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.gpt import GPTModel


class TestGPTModel(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 20
        self.context_length = 16
        self.embedding_dim = 8
        self.num_layers = 2
        self.num_heads = 2
        self.ffn_hidden_dim = 32
        self.model = GPTModel(
            self.vocab_size,
            self.context_length,
            self.embedding_dim,
            self.num_layers,
            self.num_heads,
            self.ffn_hidden_dim,
            dropout=0.0,
        )
        self.model.eval()

    def test_forward_pass_output_shape(self):
        token_ids = torch.randint(0, self.vocab_size, (4, 10))
        logits = self.model(token_ids)
        self.assertEqual(logits.shape, (4, 10, self.vocab_size))

    def test_random_input_full_context_length(self):
        token_ids = torch.randint(0, self.vocab_size, (2, self.context_length))
        logits = self.model(token_ids)
        self.assertEqual(logits.shape, (2, self.context_length, self.vocab_size))
        self.assertTrue(torch.isfinite(logits).all())

    def test_causality_preserved_end_to_end(self):
        torch.manual_seed(0)
        token_ids = torch.randint(0, self.vocab_size, (1, 6))
        logits_before = self.model(token_ids)

        token_ids_modified = token_ids.clone()
        token_ids_modified[0, -1] = (token_ids[0, -1] + 1) % self.vocab_size
        logits_after = self.model(token_ids_modified)

        self.assertTrue(torch.allclose(logits_before[0, :-1], logits_after[0, :-1], atol=1e-5))

    def test_raises_when_sequence_exceeds_context_length(self):
        token_ids = torch.randint(0, self.vocab_size, (1, self.context_length + 1))
        with self.assertRaises(ValueError):
            self.model(token_ids)

    def test_gradients_flow_to_every_component(self):
        model = GPTModel(
            self.vocab_size, self.context_length, self.embedding_dim,
            self.num_layers, self.num_heads, self.ffn_hidden_dim, dropout=0.0,
        )
        token_ids = torch.randint(0, self.vocab_size, (2, 5))
        loss = model(token_ids).sum()
        loss.backward()
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"{name} got no gradient")

    def test_num_parameters_matches_manual_sum(self):
        manual_total = sum(p.numel() for p in self.model.parameters())
        self.assertEqual(self.model.num_parameters(), manual_total)

    def test_summary_contains_component_breakdown(self):
        text = self.model.summary()
        for expected in ["embedding", "blocks", "final_ln", "lm_head", "TOTAL"]:
            self.assertIn(expected, text)

    def test_from_config_builds_from_model_config_json(self):
        config = json.loads((ROOT / "configs" / "model_config.json").read_text(encoding="utf-8"))
        model = GPTModel.from_config(config)
        self.assertEqual(model.config["vocab_size"], config["vocab_size"])
        token_ids = torch.randint(0, config["vocab_size"], (1, 8))
        logits = model(token_ids)
        self.assertEqual(logits.shape, (1, 8, config["vocab_size"]))


if __name__ == "__main__":
    unittest.main()
