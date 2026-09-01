import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.gpt import GPTModel
from training.dataset import TextDataset
from training.train import evaluate, train


def make_tiny_model(vocab_size=10, context_length=4):
    return GPTModel(
        vocab_size, context_length, embedding_dim=8,
        num_layers=2, num_heads=2, ffn_hidden_dim=16, dropout=0.0,
    )


def make_loader(vocab_size, context_length, num_sequences, batch_size, shuffle=False):
    ids = torch.randint(0, vocab_size, (context_length + num_sequences,)).tolist()
    ds = TextDataset(ids, context_length)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


class TestEvaluate(unittest.TestCase):
    def test_returns_average_loss_and_restores_train_mode(self):
        torch.manual_seed(0)
        model = make_tiny_model()
        model.train()
        loader = make_loader(10, 4, num_sequences=6, batch_size=2)
        val_loss = evaluate(model, loader)
        self.assertIsInstance(val_loss, float)
        self.assertTrue(model.training)


class TestTrainSmoke(unittest.TestCase):
    def test_smoke_run_produces_finite_loss_history(self):
        torch.manual_seed(0)
        model = make_tiny_model()
        train_loader = make_loader(10, 4, num_sequences=20, batch_size=4, shuffle=True)
        val_loader = make_loader(10, 4, num_sequences=8, batch_size=4)
        config = {
            "learning_rate": 3e-3, "weight_decay": 0.0, "epochs": 3,
            "grad_accumulation_steps": 1, "grad_clip_norm": 1.0,
        }

        history = train(model, train_loader, val_loader, config, eval_every=2, log_fn=lambda m: None)

        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in history["train_loss"]))
        self.assertGreater(len(history["train_loss"]), 0)
        self.assertGreater(len(history["val_loss"]), 0)


class TestGradAccumulation(unittest.TestCase):
    def test_optimizer_step_count_matches_accumulation_steps(self):
        # 8 sequences, batch_size=1 -> 8 batches/epoch. Counting entries in
        # history["train_loss"] counts optimizer.step() calls exactly, since
        # our loop appends once per step, never per batch.
        torch.manual_seed(0)
        config_base = {"learning_rate": 1e-3, "weight_decay": 0.0, "epochs": 1, "grad_clip_norm": None}

        model_no_accum = make_tiny_model()
        loader_no_accum = make_loader(10, 4, num_sequences=8, batch_size=1)
        history_no_accum = train(
            model_no_accum, loader_no_accum, loader_no_accum,
            {**config_base, "grad_accumulation_steps": 1}, log_fn=lambda m: None,
        )

        model_accum = make_tiny_model()
        loader_accum = make_loader(10, 4, num_sequences=8, batch_size=1)
        history_accum = train(
            model_accum, loader_accum, loader_accum,
            {**config_base, "grad_accumulation_steps": 4}, log_fn=lambda m: None,
        )

        self.assertEqual(len(history_no_accum["train_loss"]), 8)  # one step per batch
        self.assertEqual(len(history_accum["train_loss"]), 2)     # one step per 4 batches


class TestGradClipping(unittest.TestCase):
    """Note: we test that clipping is correctly WIRED (called with the right
    threshold, at the right point in the loop) rather than testing its
    effect on AdamW's output magnitude. Adam adaptively normalizes updates
    by a running gradient-variance estimate, so its step size is already
    close to scale-invariant to the raw gradient's magnitude -- clipping
    vs. not clipping would NOT reliably produce a detectably different
    update through Adam, even though the clipping call itself is correct.
    Testing the call, not an unreliable downstream proxy for it."""

    def test_clip_grad_norm_called_with_configured_threshold(self):
        torch.manual_seed(0)
        model = make_tiny_model()
        loader = make_loader(10, 4, num_sequences=4, batch_size=4)
        config = {
            "learning_rate": 1e-3, "weight_decay": 0.0, "epochs": 1,
            "grad_accumulation_steps": 1, "grad_clip_norm": 0.5,
        }

        with patch("torch.nn.utils.clip_grad_norm_") as mock_clip:
            train(model, loader, loader, config, log_fn=lambda m: None)

        mock_clip.assert_called_once()
        args, kwargs = mock_clip.call_args
        called_max_norm = kwargs.get("max_norm", args[1] if len(args) > 1 else None)
        self.assertEqual(called_max_norm, 0.5)

    def test_clip_grad_norm_not_called_when_disabled(self):
        torch.manual_seed(0)
        model = make_tiny_model()
        loader = make_loader(10, 4, num_sequences=4, batch_size=4)
        config = {
            "learning_rate": 1e-3, "weight_decay": 0.0, "epochs": 1,
            "grad_accumulation_steps": 1, "grad_clip_norm": None,
        }

        with patch("torch.nn.utils.clip_grad_norm_") as mock_clip:
            train(model, loader, loader, config, log_fn=lambda m: None)

        mock_clip.assert_not_called()

    def test_clipping_not_applied_before_accumulation_boundary(self):
        # With accumulation_steps=3 and only 2 batches available, the
        # boundary is never reached within the epoch, so clipping (and the
        # optimizer step it precedes) must never fire.
        torch.manual_seed(0)
        model = make_tiny_model()
        loader = make_loader(10, 4, num_sequences=2, batch_size=1)
        config = {
            "learning_rate": 1e-3, "weight_decay": 0.0, "epochs": 1,
            "grad_accumulation_steps": 3, "grad_clip_norm": 0.5,
        }

        with patch("torch.nn.utils.clip_grad_norm_") as mock_clip:
            history = train(model, loader, loader, config, log_fn=lambda m: None)

        mock_clip.assert_not_called()
        self.assertEqual(len(history["train_loss"]), 0)


class TestCheckpointResumeIntegration(unittest.TestCase):
    def test_resume_continues_step_count_and_history(self):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "ckpt.pt")
            model = make_tiny_model()
            loader = make_loader(10, 4, num_sequences=8, batch_size=2)
            config = {
                "learning_rate": 1e-3, "weight_decay": 0.0, "epochs": 5,
                "grad_accumulation_steps": 1, "grad_clip_norm": 1.0,
            }

            first_history = train(model, loader, loader, config, max_steps=3, checkpoint_path=path, log_fn=lambda m: None)
            self.assertEqual(len(first_history["train_loss"]), 3)

            resumed_model = make_tiny_model()
            second_history = train(
                resumed_model, loader, loader, config,
                max_steps=6, resume_from=path, checkpoint_path=path, log_fn=lambda m: None,
            )
            self.assertEqual(len(second_history["train_loss"]), 6)  # 3 restored + 3 new


if __name__ == "__main__":
    unittest.main()
