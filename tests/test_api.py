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

TEST_API_KEY = "test-secret-key"


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
            {
                "CHECKPOINT_PATH": self.checkpoint_path,
                "TOKENIZER_PATH": self.tokenizer_path,
                "API_KEY": TEST_API_KEY,
            },
        )
        self.env_patch.start()

        # api.main is imported once per process (Python caches modules), so
        # CORS/middleware config (fixed at import time) always reflects the
        # FIRST import's environment -- but lifespan() re-reads env vars
        # fresh on every TestClient startup, so model/api-key/rate-limit
        # config IS reliably per-test via env patching.
        import api.main as api_main
        self.api_main = api_main
        self.auth_headers = {"X-API-Key": TEST_API_KEY}

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

    def test_health_requires_no_api_key(self):
        with TestClient(self.api_main.app) as client:
            response = client.get("/health")  # no auth header at all
        self.assertEqual(response.status_code, 200)

    def test_generate_returns_text_for_a_valid_authenticated_request(self):
        with TestClient(self.api_main.app) as client:
            response = client.post(
                "/generate", json={"prompt": "the quick", "max_new_tokens": 10, "greedy": True},
                headers=self.auth_headers,
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("text", body)
        self.assertTrue(body["text"].startswith("the quick"))

    def test_generate_is_deterministic_with_greedy(self):
        with TestClient(self.api_main.app) as client:
            r1 = client.post("/generate", json={"prompt": "dog", "max_new_tokens": 15, "greedy": True}, headers=self.auth_headers)
            r2 = client.post("/generate", json={"prompt": "dog", "max_new_tokens": 15, "greedy": True}, headers=self.auth_headers)
        self.assertEqual(r1.json()["text"], r2.json()["text"])

    def test_rejects_empty_prompt(self):
        with TestClient(self.api_main.app) as client:
            response = client.post("/generate", json={"prompt": ""}, headers=self.auth_headers)
        self.assertEqual(response.status_code, 422)

    def test_rejects_max_new_tokens_out_of_range(self):
        with TestClient(self.api_main.app) as client:
            too_many = client.post("/generate", json={"prompt": "the", "max_new_tokens": 5000}, headers=self.auth_headers)
            zero = client.post("/generate", json={"prompt": "the", "max_new_tokens": 0}, headers=self.auth_headers)
        self.assertEqual(too_many.status_code, 422)
        self.assertEqual(zero.status_code, 422)

    def test_rejects_non_positive_temperature(self):
        with TestClient(self.api_main.app) as client:
            response = client.post("/generate", json={"prompt": "the", "temperature": 0}, headers=self.auth_headers)
        self.assertEqual(response.status_code, 422)

    def test_rejects_top_p_out_of_range(self):
        with TestClient(self.api_main.app) as client:
            response = client.post("/generate", json={"prompt": "the", "top_p": 1.5}, headers=self.auth_headers)
        self.assertEqual(response.status_code, 422)

    def test_generation_failure_returns_500_without_leaking_internals(self):
        with TestClient(self.api_main.app) as client:
            with patch("api.main.generate_text", side_effect=RuntimeError("some internal detail")):
                response = client.post("/generate", json={"prompt": "the"}, headers=self.auth_headers)
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("some internal detail", response.text)

    def test_unsupported_device_fails_startup_clearly(self):
        with patch.dict(os.environ, {"DEVICE": "cuda"}):
            with self.assertRaises(RuntimeError):
                with TestClient(self.api_main.app):
                    pass

    # --- Phase 19: auth ---

    def test_generate_rejects_missing_api_key(self):
        with TestClient(self.api_main.app) as client:
            response = client.post("/generate", json={"prompt": "the"})  # no header
        self.assertEqual(response.status_code, 401)

    def test_generate_rejects_wrong_api_key(self):
        with TestClient(self.api_main.app) as client:
            response = client.post("/generate", json={"prompt": "the"}, headers={"X-API-Key": "wrong-key"})
        self.assertEqual(response.status_code, 401)

    def test_generate_fails_closed_when_server_has_no_api_key_configured(self):
        with patch.dict(os.environ, {"API_KEY": ""}):
            with TestClient(self.api_main.app) as client:
                # even the "no key sent" and "correct-looking key" cases must
                # both fail when the server itself has nothing configured
                response = client.post("/generate", json={"prompt": "the"}, headers=self.auth_headers)
        self.assertEqual(response.status_code, 503)

    # --- Phase 19: rate limiting ---

    def test_rate_limit_blocks_requests_over_the_configured_threshold(self):
        with patch.dict(os.environ, {"RATE_LIMIT_REQUESTS": "3", "RATE_LIMIT_WINDOW_SECONDS": "60"}):
            with TestClient(self.api_main.app) as client:
                statuses = [
                    client.post("/generate", json={"prompt": "the", "max_new_tokens": 1, "greedy": True}, headers=self.auth_headers).status_code
                    for _ in range(4)
                ]
        self.assertEqual(statuses[:3], [200, 200, 200])
        self.assertEqual(statuses[3], 429)

    def test_rate_limit_is_isolated_per_client(self):
        with patch.dict(os.environ, {"RATE_LIMIT_REQUESTS": "1", "RATE_LIMIT_WINDOW_SECONDS": "60"}):
            with TestClient(self.api_main.app, client=("1.2.3.4", 123)) as client_a, \
                 TestClient(self.api_main.app, client=("5.6.7.8", 456)) as client_b:
                r1 = client_a.post("/generate", json={"prompt": "the", "max_new_tokens": 1, "greedy": True}, headers=self.auth_headers)
                r2 = client_b.post("/generate", json={"prompt": "the", "max_new_tokens": 1, "greedy": True}, headers=self.auth_headers)
        # NOTE: these share the same underlying app.state.rate_limiter instance
        # (module-cached app), so this exercises per-client-IP isolation, not
        # per-app isolation -- both should succeed since they're different IPs.
        self.assertEqual(r1.status_code, 200)
        self.assertEqual(r2.status_code, 200)

    # --- Phase 19: request size limit ---

    def test_rejects_oversized_request_body(self):
        huge_prompt = "a" * 20_000  # exceeds the 10,000 byte middleware limit
        with TestClient(self.api_main.app) as client:
            response = client.post("/generate", json={"prompt": huge_prompt}, headers=self.auth_headers)
        self.assertEqual(response.status_code, 413)

    # --- Phase 19: CORS (fixed at module-import time -- see setUp note) ---

    def test_cors_denies_by_default_for_an_unlisted_origin(self):
        with TestClient(self.api_main.app) as client:
            response = client.get("/health", headers={"Origin": "http://evil.example.com"})
        self.assertNotIn("access-control-allow-origin", {k.lower() for k in response.headers.keys()})


class TestParseAllowedOrigins(unittest.TestCase):
    def test_empty_or_none_gives_no_origins(self):
        from api.security import parse_allowed_origins
        self.assertEqual(parse_allowed_origins(""), [])
        self.assertEqual(parse_allowed_origins(None), [])

    def test_splits_and_strips_comma_separated_origins(self):
        from api.security import parse_allowed_origins
        self.assertEqual(
            parse_allowed_origins(" http://localhost:5173 , http://example.com "),
            ["http://localhost:5173", "http://example.com"],
        )


class TestRateLimiter(unittest.TestCase):
    def test_allows_up_to_the_limit_then_blocks(self):
        from api.security import RateLimiter
        limiter = RateLimiter(limit=2, window_seconds=60)
        self.assertTrue(limiter.is_allowed("client-a"))
        self.assertTrue(limiter.is_allowed("client-a"))
        self.assertFalse(limiter.is_allowed("client-a"))

    def test_different_clients_have_independent_limits(self):
        from api.security import RateLimiter
        limiter = RateLimiter(limit=1, window_seconds=60)
        self.assertTrue(limiter.is_allowed("client-a"))
        self.assertTrue(limiter.is_allowed("client-b"))
        self.assertFalse(limiter.is_allowed("client-a"))

    def test_old_requests_outside_the_window_are_forgotten(self):
        from api.security import RateLimiter
        limiter = RateLimiter(limit=1, window_seconds=60)
        limiter.is_allowed("client-a")
        # simulate the window having fully elapsed
        limiter._requests["client-a"][0] -= 61
        self.assertTrue(limiter.is_allowed("client-a"))


if __name__ == "__main__":
    unittest.main()
