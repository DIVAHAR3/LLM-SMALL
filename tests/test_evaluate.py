import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.gpt import GPTModel
from tokenizer.char_tokenizer import CharTokenizer
from training.checkpoint import save_checkpoint
from training.dataset import TextDataset
from training.evaluate import compare_checkpoints, evaluate_checkpoint, format_report, perplexity


MODEL_CONFIG = {
    # vocab_size must comfortably cover any tokenizer built from this
    # file's sample corpora (~27 unique chars + 4 special tokens) so
    # generated/sampled token ids never overflow the embedding table.
    "vocab_size": 40, "context_length": 6, "embedding_dim": 8,
    "num_layers": 2, "num_heads": 2, "ffn_hidden_dim": 16, "dropout": 0.0,
}


def make_checkpoint(path, epoch, step, extra_train_loss=None):
    torch.manual_seed(0)
    model = GPTModel.from_config(MODEL_CONFIG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    metrics = {"train_loss": extra_train_loss or [2.0, 1.9, 1.8], "val_loss": [2.1, 1.95]}
    save_checkpoint(path, model, optimizer, epoch=epoch, step=step, config={}, metrics=metrics)


class TestPerplexity(unittest.TestCase):
    def test_zero_loss_gives_perplexity_one(self):
        self.assertAlmostEqual(perplexity(0.0), 1.0)

    def test_matches_the_untrained_baseline_from_phase_11(self):
        vocab_size = 51
        self.assertAlmostEqual(perplexity(math.log(vocab_size)), vocab_size, places=5)


class TestEvaluateCheckpoint(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.ckpt_path = str(Path(self.tmpdir.name) / "ckpt.pt")
        make_checkpoint(self.ckpt_path, epoch=1, step=10)

        corpus = "the quick brown fox jumps over the lazy dog " * 3
        self.tokenizer = CharTokenizer.from_text(corpus)
        torch.manual_seed(1)
        self.val_ids = torch.randint(
            0, MODEL_CONFIG["vocab_size"], (MODEL_CONFIG["context_length"] + 12,)
        ).tolist()  # 12 examples worth, valid token ids only

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_report_contains_expected_keys_and_sane_values(self):
        report = evaluate_checkpoint(self.ckpt_path, MODEL_CONFIG, self.tokenizer, self.val_ids)

        expected_keys = {
            "checkpoint_path", "epoch", "step", "param_count", "val_loss", "perplexity",
            "tokens_processed", "eval_seconds", "tokens_per_sec", "peak_python_object_memory_mb",
            "train_loss_history_tail", "val_loss_history_tail", "samples",
        }
        self.assertEqual(set(report.keys()), expected_keys)
        self.assertEqual(report["epoch"], 1)
        self.assertEqual(report["step"], 10)
        self.assertTrue(math.isfinite(report["val_loss"]))
        self.assertGreater(report["val_loss"], 0)

    def test_perplexity_is_consistent_with_reported_val_loss(self):
        report = evaluate_checkpoint(self.ckpt_path, MODEL_CONFIG, self.tokenizer, self.val_ids)
        self.assertAlmostEqual(report["perplexity"], math.exp(report["val_loss"]), places=5)

    def test_param_count_matches_a_freshly_constructed_model(self):
        report = evaluate_checkpoint(self.ckpt_path, MODEL_CONFIG, self.tokenizer, self.val_ids)
        fresh_model = GPTModel.from_config(MODEL_CONFIG)
        self.assertEqual(report["param_count"], fresh_model.num_parameters())

    def test_tokens_processed_matches_full_val_set_coverage(self):
        report = evaluate_checkpoint(self.ckpt_path, MODEL_CONFIG, self.tokenizer, self.val_ids)
        context_length = MODEL_CONFIG["context_length"]
        expected = len(TextDataset(self.val_ids, context_length)) * context_length
        self.assertEqual(report["tokens_processed"], expected)

    def test_no_samples_when_prompts_not_requested(self):
        report = evaluate_checkpoint(self.ckpt_path, MODEL_CONFIG, self.tokenizer, self.val_ids, sample_prompts=None)
        self.assertEqual(report["samples"], [])

    def test_samples_generated_when_prompts_requested(self):
        report = evaluate_checkpoint(
            self.ckpt_path, MODEL_CONFIG, self.tokenizer, self.val_ids,
            sample_prompts=["the quick", "dog"], max_new_tokens=5,
        )
        self.assertEqual(len(report["samples"]), 2)
        self.assertEqual(report["samples"][0]["prompt"], "the quick")
        self.assertTrue(report["samples"][0]["generated"].startswith("the quick"))

    def test_history_tails_come_from_the_checkpoints_own_metrics(self):
        make_checkpoint(self.ckpt_path, epoch=1, step=10, extra_train_loss=[3.0, 2.5, 2.0, 1.5, 1.0, 0.5])
        report = evaluate_checkpoint(self.ckpt_path, MODEL_CONFIG, self.tokenizer, self.val_ids)
        # only the last 5 of the 6 logged entries should be kept
        self.assertEqual(report["train_loss_history_tail"], [2.5, 2.0, 1.5, 1.0, 0.5])


class TestFormatReport(unittest.TestCase):
    def test_report_text_includes_key_metrics(self):
        report = {
            "checkpoint_path": "ckpt.pt", "epoch": 2, "step": 50, "param_count": 12345,
            "val_loss": 1.2345, "perplexity": 3.4363, "tokens_processed": 1000,
            "eval_seconds": 0.5, "tokens_per_sec": 2000.0, "peak_python_object_memory_mb": 1.5,
            "train_loss_history_tail": [1.1, 1.0], "val_loss_history_tail": [1.3],
            "samples": [{"prompt": "hi", "generated": "hi there"}],
        }
        text = format_report(report)
        self.assertIn("ckpt.pt", text)
        self.assertIn("1.2345", text)
        self.assertIn("3.44", text)  # perplexity rounded to 2 places in the format string
        self.assertIn("hi there", text)


class TestCompareCheckpoints(unittest.TestCase):
    def test_comparison_mentions_both_checkpoints(self):
        with tempfile.TemporaryDirectory() as d:
            path_a = str(Path(d) / "a.pt")
            path_b = str(Path(d) / "b.pt")
            make_checkpoint(path_a, epoch=0, step=5)
            make_checkpoint(path_b, epoch=1, step=20)

            corpus = "the quick brown fox " * 3
            tokenizer = CharTokenizer.from_text(corpus)
            val_ids = list(range(MODEL_CONFIG["context_length"] + 8))

            report_a = evaluate_checkpoint(path_a, MODEL_CONFIG, tokenizer, val_ids)
            report_b = evaluate_checkpoint(path_b, MODEL_CONFIG, tokenizer, val_ids)

            text = compare_checkpoints([report_a, report_b])
            self.assertIn("a.pt", text)
            self.assertIn("b.pt", text)
            self.assertIn("5", text)
            self.assertIn("20", text)


if __name__ == "__main__":
    unittest.main()
