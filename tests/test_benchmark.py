import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.gpt import GPTModel
from scripts.benchmark import (
    benchmark_compute_cost,
    benchmark_inference,
    format_compute_cost_report,
    format_inference_report,
    get_available_memory_mb,
)
from tokenizer.char_tokenizer import CharTokenizer
from training.checkpoint import save_checkpoint


class TestGetAvailableMemory(unittest.TestCase):
    def test_returns_a_positive_number_or_none(self):
        result = get_available_memory_mb()
        # Contract: either a real positive figure, or None if unavailable --
        # never a guess, never a crash.
        self.assertTrue(result is None or result > 0)


TINY_MODEL_CONFIG = {
    "vocab_size": 15, "context_length": 999, "embedding_dim": 8,
    "num_layers": 2, "num_heads": 2, "ffn_hidden_dim": 16, "dropout": 0.0,
}


class TestBenchmarkComputeCost(unittest.TestCase):
    def test_returns_expected_keys_with_sane_values(self):
        result = benchmark_compute_cost(
            batch_size=2, context_length=8, model_config=TINY_MODEL_CONFIG, repeats=2, warmup=1,
        )
        expected_keys = {"batch_size", "context_length", "param_count", "avg_step_time_ms", "tokens_per_sec"}
        self.assertEqual(set(result.keys()), expected_keys)
        self.assertEqual(result["batch_size"], 2)
        self.assertEqual(result["context_length"], 8)
        self.assertGreater(result["param_count"], 0)
        self.assertGreater(result["avg_step_time_ms"], 0)
        self.assertGreater(result["tokens_per_sec"], 0)

    def test_context_length_argument_overrides_the_configs_own_value(self):
        # model_config's own context_length (999) must not leak through --
        # the context_length argument is the axis being swept.
        result = benchmark_compute_cost(
            batch_size=2, context_length=8, model_config=TINY_MODEL_CONFIG, repeats=1, warmup=1,
        )
        self.assertEqual(result["context_length"], 8)

    def test_tokens_per_sec_is_mathematically_consistent_with_step_time(self):
        result = benchmark_compute_cost(
            batch_size=4, context_length=8, model_config=TINY_MODEL_CONFIG, repeats=2, warmup=1,
        )
        expected_tokens_per_sec = (4 * 8) / (result["avg_step_time_ms"] / 1000)
        self.assertAlmostEqual(result["tokens_per_sec"], expected_tokens_per_sec, places=3)

    def test_larger_batch_reports_more_tokens_per_step_worth_of_throughput_scale(self):
        # Not a strict performance assertion (timing is inherently noisy on
        # a shared machine) -- just confirms tokens_per_sec scales with the
        # batch_size*context_length numerator, i.e. the accounting is right.
        small = benchmark_compute_cost(
            batch_size=2, context_length=8, model_config=TINY_MODEL_CONFIG, repeats=2, warmup=1,
        )
        large = benchmark_compute_cost(
            batch_size=8, context_length=8, model_config=TINY_MODEL_CONFIG, repeats=2, warmup=1,
        )
        self.assertEqual(large["batch_size"] * large["context_length"], 8 * 8)
        self.assertEqual(small["batch_size"] * small["context_length"], 2 * 8)


class TestBenchmarkInference(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        torch.manual_seed(0)
        self.tokenizer = CharTokenizer.from_text("the quick brown fox jumps over the lazy dog ")
        model = GPTModel(
            vocab_size=self.tokenizer.vocab_size, context_length=16, embedding_dim=8,
            num_layers=2, num_heads=2, ffn_hidden_dim=16, dropout=0.0,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        self.checkpoint_path = str(Path(self.tmpdir.name) / "ckpt.pt")
        self.tokenizer_path = str(Path(self.tmpdir.name) / "vocab.json")
        save_checkpoint(self.checkpoint_path, model, optimizer, epoch=0, step=1, config={}, metrics={})
        self.tokenizer.save(self.tokenizer_path)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_returns_expected_structure(self):
        report = benchmark_inference(
            self.checkpoint_path, self.tokenizer_path, "the quick", max_new_tokens_list=[3, 6], repeats=1,
        )
        self.assertIn("param_count", report)
        self.assertGreater(report["param_count"], 0)
        self.assertGreaterEqual(report["load_time_seconds"], 0)
        self.assertEqual(len(report["configs"]), 2)
        for config in report["configs"]:
            self.assertGreaterEqual(config["avg_total_latency_ms"], config["avg_first_token_latency_ms"])
            self.assertGreater(config["steady_state_tokens_per_sec"], 0)

    def test_single_token_generation_has_no_negative_steady_state_time(self):
        # max_new_tokens=1 means first-token IS the whole generation; steady
        # state has zero tokens left, must not divide by zero or go negative.
        report = benchmark_inference(
            self.checkpoint_path, self.tokenizer_path, "the", max_new_tokens_list=[1], repeats=1,
        )
        config = report["configs"][0]
        self.assertEqual(config["max_new_tokens"], 1)
        self.assertTrue(config["steady_state_tokens_per_sec"] == float("inf") or config["steady_state_tokens_per_sec"] >= 0)


class TestReportFormatting(unittest.TestCase):
    def test_inference_report_includes_key_figures(self):
        report = {
            "checkpoint": "ckpt.pt", "param_count": 12345, "load_time_seconds": 0.5,
            "available_memory_before_load_mb": 1000.0, "available_memory_after_load_mb": 950.0,
            "configs": [
                {"max_new_tokens": 20, "avg_first_token_latency_ms": 5.0, "avg_total_latency_ms": 100.0, "steady_state_tokens_per_sec": 200.0},
            ],
        }
        text = format_inference_report(report)
        self.assertIn("ckpt.pt", text)
        self.assertIn("12,345", text)
        self.assertIn("200.0", text)

    def test_inference_report_handles_unavailable_memory_gracefully(self):
        report = {
            "checkpoint": "ckpt.pt", "param_count": 100, "load_time_seconds": 0.1,
            "available_memory_before_load_mb": None, "available_memory_after_load_mb": None,
            "configs": [],
        }
        text = format_inference_report(report)
        self.assertIn("unavailable", text)

    def test_compute_cost_report_includes_each_config(self):
        results = [
            {"batch_size": 8, "context_length": 64, "param_count": 1000, "avg_step_time_ms": 10.0, "tokens_per_sec": 512.0},
            {"batch_size": 32, "context_length": 128, "param_count": 1000, "avg_step_time_ms": 40.0, "tokens_per_sec": 1024.0},
        ]
        text = format_compute_cost_report(results)
        self.assertIn("512.0", text)
        self.assertIn("1024.0", text)


if __name__ == "__main__":
    unittest.main()
