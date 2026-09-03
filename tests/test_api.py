import json
import os
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import torch
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.gpt import GPTModel
from ocr.model import CharacterCNN
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


def make_tiny_ocr_checkpoint(tmpdir):
    torch.manual_seed(0)
    model = CharacterCNN(image_size=28, num_classes=62, conv1_channels=4, conv2_channels=8, hidden_dim=16, dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint_path = str(Path(tmpdir) / "ocr_ckpt.pt")
    save_checkpoint(checkpoint_path, model, optimizer, epoch=0, step=1, config={}, metrics={})
    return checkpoint_path


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
                # explicit, deliberately-nonexistent path: keeps tests hermetic
                # and independent of whatever OCR checkpoint may or may not
                # exist locally -- OCR-specific tests below override this.
                "OCR_CHECKPOINT_PATH": str(Path(self.tmpdir.name) / "no_such_ocr_checkpoint.pt"),
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

    # --- Phase 21: streaming ---

    def test_stream_yields_sse_events_ending_in_done(self):
        with TestClient(self.api_main.app) as client:
            response = client.post(
                "/generate/stream", json={"prompt": "the quick", "max_new_tokens": 5, "greedy": True},
                headers=self.auth_headers,
            )
        self.assertEqual(response.status_code, 200)
        events = parse_sse_events(response.text)
        self.assertGreater(len(events), 0)
        self.assertEqual(events[-1], {"done": True})
        for event in events[:-1]:
            self.assertIn("chunk", event)

    def test_stream_reconstructs_the_same_text_as_non_streaming_generation(self):
        payload = {"prompt": "the quick", "max_new_tokens": 8, "greedy": True}
        with TestClient(self.api_main.app) as client:
            non_streaming = client.post("/generate", json=payload, headers=self.auth_headers).json()["text"]
            streamed = client.post("/generate/stream", json=payload, headers=self.auth_headers)

        events = parse_sse_events(streamed.text)
        streamed_text = payload["prompt"] + "".join(e["chunk"] for e in events if "chunk" in e)
        self.assertEqual(streamed_text, non_streaming)

    def test_stream_rejects_missing_api_key(self):
        with TestClient(self.api_main.app) as client:
            response = client.post("/generate/stream", json={"prompt": "the"})
        self.assertEqual(response.status_code, 401)

    def test_stream_respects_rate_limit(self):
        with patch.dict(os.environ, {"RATE_LIMIT_REQUESTS": "1", "RATE_LIMIT_WINDOW_SECONDS": "60"}):
            with TestClient(self.api_main.app) as client:
                first = client.post(
                    "/generate/stream", json={"prompt": "the", "max_new_tokens": 1, "greedy": True},
                    headers=self.auth_headers,
                )
                second = client.post(
                    "/generate/stream", json={"prompt": "the", "max_new_tokens": 1, "greedy": True},
                    headers=self.auth_headers,
                )
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 429)

    def test_stream_rejects_invalid_request_before_streaming_starts(self):
        with TestClient(self.api_main.app) as client:
            response = client.post("/generate/stream", json={"prompt": ""}, headers=self.auth_headers)
        self.assertEqual(response.status_code, 422)

    # --- Image analysis (classical, no ML) ---

    def test_analyze_image_returns_expected_shape_for_a_valid_image(self):
        image = Image.new("RGB", (20, 10), color=(10, 20, 30))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        with TestClient(self.api_main.app) as client:
            response = client.post(
                "/analyze/image",
                files={"file": ("test.png", buffer.getvalue(), "image/png")},
                headers=self.auth_headers,
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["width"], 20)
        self.assertEqual(body["height"], 10)
        self.assertEqual(body["format"], "PNG")
        self.assertEqual(len(body["dominant_colors"]), 1)
        self.assertEqual(body["dominant_colors"][0]["hex"], "#0a141e")

    def test_analyze_image_omits_ocr_text_when_ocr_checkpoint_is_unavailable(self):
        # setUp points OCR_CHECKPOINT_PATH at a nonexistent file by default
        image = Image.new("RGB", (10, 10), color=(1, 2, 3))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        with TestClient(self.api_main.app) as client:
            response = client.post(
                "/analyze/image", files={"file": ("t.png", buffer.getvalue(), "image/png")}, headers=self.auth_headers,
            )
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["ocr_text"])

    def test_analyze_image_includes_ocr_text_when_a_checkpoint_is_available(self):
        ocr_checkpoint_path = make_tiny_ocr_checkpoint(self.tmpdir.name)
        image = Image.new("RGB", (40, 20), color=(200, 200, 200))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        with patch.dict(os.environ, {"OCR_CHECKPOINT_PATH": ocr_checkpoint_path}):
            with TestClient(self.api_main.app) as client:
                response = client.post(
                    "/analyze/image", files={"file": ("t.png", buffer.getvalue(), "image/png")}, headers=self.auth_headers,
                )
        self.assertEqual(response.status_code, 200)
        # a blank 40x20 image has no ink -- correctly extracts to empty text,
        # not None, proving the OCR pipeline actually ran rather than being skipped
        self.assertEqual(response.json()["ocr_text"], "")

    def test_analyze_image_survives_an_ocr_extraction_failure(self):
        ocr_checkpoint_path = make_tiny_ocr_checkpoint(self.tmpdir.name)
        image = Image.new("RGB", (10, 10), color=(5, 5, 5))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        with patch.dict(os.environ, {"OCR_CHECKPOINT_PATH": ocr_checkpoint_path}):
            with TestClient(self.api_main.app) as client:
                with patch("api.main.extract_text", side_effect=RuntimeError("boom")):
                    response = client.post(
                        "/analyze/image", files={"file": ("t.png", buffer.getvalue(), "image/png")}, headers=self.auth_headers,
                    )
        # the classical analysis (already computed before OCR runs) must
        # still come through even though OCR itself blew up
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIsNone(body["ocr_text"])
        self.assertEqual(body["width"], 10)

    def test_analyze_image_rejects_non_image_bytes(self):
        with TestClient(self.api_main.app) as client:
            response = client.post(
                "/analyze/image",
                files={"file": ("not-an-image.png", b"this is not an image", "image/png")},
                headers=self.auth_headers,
            )
        self.assertEqual(response.status_code, 400)

    def test_analyze_image_rejects_missing_api_key(self):
        image = Image.new("RGB", (5, 5), color=(0, 0, 0))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        with TestClient(self.api_main.app) as client:
            response = client.post("/analyze/image", files={"file": ("t.png", buffer.getvalue(), "image/png")})
        self.assertEqual(response.status_code, 401)

    def test_analyze_image_respects_rate_limit(self):
        image = Image.new("RGB", (5, 5), color=(0, 0, 0))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        with patch.dict(os.environ, {"RATE_LIMIT_REQUESTS": "1", "RATE_LIMIT_WINDOW_SECONDS": "60"}):
            with TestClient(self.api_main.app) as client:
                first = client.post(
                    "/analyze/image", files={"file": ("t.png", buffer.getvalue(), "image/png")}, headers=self.auth_headers,
                )
                second = client.post(
                    "/analyze/image", files={"file": ("t.png", buffer.getvalue(), "image/png")}, headers=self.auth_headers,
                )
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 429)

    def test_analyze_image_rejects_oversized_upload(self):
        # exceeds the RequestSizeLimitMiddleware override for /analyze/image
        huge_payload = b"\x00" * (self.api_main.IMAGE_MAX_BYTES + 1)
        with TestClient(self.api_main.app) as client:
            response = client.post(
                "/analyze/image", files={"file": ("t.png", huge_payload, "image/png")}, headers=self.auth_headers,
            )
        self.assertEqual(response.status_code, 413)

    def test_analyze_image_allows_upload_larger_than_the_default_json_limit(self):
        # a real photo easily exceeds /generate's 10,000-byte default limit
        # but must still be accepted here, under the higher override
        image = Image.new("RGB", (300, 300), color=(50, 100, 150))
        buffer = BytesIO()
        image.save(buffer, format="BMP")  # uncompressed -- reliably > 10,000 bytes
        self.assertGreater(len(buffer.getvalue()), 10_000)
        with TestClient(self.api_main.app) as client:
            response = client.post(
                "/analyze/image", files={"file": ("t.bmp", buffer.getvalue(), "image/bmp")}, headers=self.auth_headers,
            )
        self.assertEqual(response.status_code, 200)

    def test_stream_emits_error_event_without_leaking_internals_on_failure(self):
        # A real stream_ids(...) call never raises synchronously -- it's a
        # generator function, so its body (and any error inside it) only
        # runs once iteration begins. Simulate that realistically: a
        # generator that yields one token successfully, then fails
        # mid-stream (the actual failure shape this needs to handle).
        def broken_stream(*args, **kwargs):
            yield 1
            raise RuntimeError("some internal detail")

        with TestClient(self.api_main.app) as client:
            with patch("api.sse.stream_ids", side_effect=broken_stream):
                response = client.post("/generate/stream", json={"prompt": "the"}, headers=self.auth_headers)
        events = parse_sse_events(response.text)
        self.assertIn("chunk", events[0])  # the successful token still made it out
        self.assertIn("error", events[-1])
        self.assertNotIn("some internal detail", response.text)


def parse_sse_events(text):
    events = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        assert block.startswith("data: "), block
        events.append(json.loads(block[len("data: "):]))
    return events


class TestFormatEvent(unittest.TestCase):
    def test_wraps_payload_as_json_with_sse_data_prefix_and_blank_line(self):
        from api.sse import format_event
        self.assertEqual(format_event({"chunk": "a"}), 'data: {"chunk": "a"}\n\n')


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


class TestRequestSizeLimitMiddlewarePathOverrides(unittest.TestCase):
    def test_unmatched_path_uses_the_default_limit(self):
        from api.security import RequestSizeLimitMiddleware
        middleware = RequestSizeLimitMiddleware(app=None, max_bytes=10_000, path_overrides={"/analyze/image": 5_000_000})
        self.assertEqual(middleware._limit_for("/generate"), 10_000)

    def test_matched_path_prefix_uses_its_override(self):
        from api.security import RequestSizeLimitMiddleware
        middleware = RequestSizeLimitMiddleware(app=None, max_bytes=10_000, path_overrides={"/analyze/image": 5_000_000})
        self.assertEqual(middleware._limit_for("/analyze/image"), 5_000_000)

    def test_no_overrides_configured_behaves_like_before(self):
        from api.security import RequestSizeLimitMiddleware
        middleware = RequestSizeLimitMiddleware(app=None, max_bytes=10_000)
        self.assertEqual(middleware._limit_for("/anything"), 10_000)


if __name__ == "__main__":
    unittest.main()
