import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.gpt import GPTModel
from tokenizer.char_tokenizer import CharTokenizer
from training.checkpoint import save_checkpoint


def make_tiny_checkpoint_and_tokenizer(tmpdir):
    torch.manual_seed(0)
    tokenizer = CharTokenizer.from_text("the quick brown fox jumps over the lazy dog ")
    model = GPTModel(
        vocab_size=tokenizer.vocab_size, context_length=16, embedding_dim=8,
        num_layers=2, num_heads=2, ffn_hidden_dim=16, dropout=0.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    checkpoint_path = str(Path(tmpdir) / "ckpt.pt")
    tokenizer_path = str(Path(tmpdir) / "vocab.json")
    save_checkpoint(checkpoint_path, model, optimizer, epoch=0, step=1, config={}, metrics={})
    tokenizer.save(tokenizer_path)
    return checkpoint_path, tokenizer_path, model.num_parameters()


class TestAPI(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.checkpoint_path, self.tokenizer_path, self.param_count = make_tiny_checkpoint_and_tokenizer(self.tmpdir.name)
        self.env_patch = patch.dict(
            os.environ,
            {"CHECKPOINT_PATH": self.checkpoint_path, "TOKENIZER_PATH": self.tokenizer_path},
        )
        self.env_patch.start()

        # Import (or re-trigger module-level state) after env vars are set,
        # since lifespan reads them fresh on every TestClient startup.
        import api.main as api_main
        self.api_main = api_main

    def tearDown(self):
        self.env_patch.stop()
        self.tmpdir.cleanup()

    def test_health_reports_configured_checkpoint_and_param_count(self):
        with TestClient(self.api_main.app) as client:
            response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["checkpoint"], self.checkpoint_path)
        self.assertEqual(body["params"], self.param_count)

    def test_generate_returns_text_for_a_valid_request(self):
        with TestClient(self.api_main.app) as client:
            response = client.post("/generate", json={"prompt": "the quick", "max_new_tokens": 10, "greedy": True})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("text", body)
        self.assertIsInstance(body["text"], str)
        self.assertTrue(body["text"].startswith("the quick"))

    def test_generate_is_deterministic_with_greedy(self):
        with TestClient(self.api_main.app) as client:
            r1 = client.post("/generate", json={"prompt": "dog", "max_new_tokens": 15, "greedy": True})
            r2 = client.post("/generate", json={"prompt": "dog", "max_new_tokens": 15, "greedy": True})
        self.assertEqual(r1.json()["text"], r2.json()["text"])

    def test_rejects_empty_prompt(self):
        with TestClient(self.api_main.app) as client:
            response = client.post("/generate", json={"prompt": ""})
        self.assertEqual(response.status_code, 422)

    def test_rejects_max_new_tokens_out_of_range(self):
        with TestClient(self.api_main.app) as client:
            too_many = client.post("/generate", json={"prompt": "the", "max_new_tokens": 5000})
            zero = client.post("/generate", json={"prompt": "the", "max_new_tokens": 0})
        self.assertEqual(too_many.status_code, 422)
        self.assertEqual(zero.status_code, 422)

    def test_rejects_non_positive_temperature(self):
        with TestClient(self.api_main.app) as client:
            response = client.post("/generate", json={"prompt": "the", "temperature": 0})
        self.assertEqual(response.status_code, 422)

    def test_rejects_top_p_out_of_range(self):
        with TestClient(self.api_main.app) as client:
            response = client.post("/generate", json={"prompt": "the", "top_p": 1.5})
        self.assertEqual(response.status_code, 422)

    def test_generation_failure_returns_500_without_leaking_internals(self):
        with TestClient(self.api_main.app) as client:
            with patch("api.main.generate_text", side_effect=RuntimeError("some internal detail")):
                response = client.post("/generate", json={"prompt": "the"})
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("some internal detail", response.text)

    def test_unsupported_device_fails_startup_clearly(self):
        with patch.dict(os.environ, {"DEVICE": "cuda"}):
            with self.assertRaises(RuntimeError):
                with TestClient(self.api_main.app):
                    pass


if __name__ == "__main__":
    unittest.main()
