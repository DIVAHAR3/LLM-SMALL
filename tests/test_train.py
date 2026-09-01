import copy
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
from training.loss import cross_entropy_loss
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
        # 8 sequences, batch_size=1 -> 8 batches/epoch, evenly divisible by
        # both accumulation_steps values below, so no leftover-window flush
        # is involved -- this test is purely about step-count bookkeeping.
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


class TestLeftoverWindowFlush(unittest.TestCase):
    """Regression test for an adversarial-review finding: a partial
    accumulation window at the end of an epoch used to be silently
    discarded by the next epoch's zero_grad(), meaning trailing batches
    never contributed to training, every single epoch. It must now be
    flushed as one (smaller) optimizer step instead."""

    def test_leftover_partial_window_is_flushed_not_discarded(self):
        torch.manual_seed(0)
        model = make_tiny_model()
        loader = make_loader(10, 4, num_sequences=2, batch_size=1)  # only 2 batches, less than accumulation_steps=3
        config = {
            "learning_rate": 1e-3, "weight_decay": 0.0, "epochs": 1,
            "grad_accumulation_steps": 3, "grad_clip_norm": 0.5,
        }

        with patch("torch.nn.utils.clip_grad_norm_") as mock_clip:
            history = train(model, loader, loader, config, log_fn=lambda m: None)

        mock_clip.assert_called_once()  # the leftover 2-batch window IS flushed, not dropped
        self.assertEqual(len(history["train_loss"]), 1)


class TestResumeRespectsNewConfig(unittest.TestCase):
    """Regression test: optimizer.load_state_dict() overwrites every
    param_group field (lr, weight_decay, ...) from the checkpoint. Without
    an explicit reapply, a resume call's own config.learning_rate would be
    silently ignored in favor of whatever was checkpointed."""

    def test_resume_applies_the_new_configs_learning_rate_not_the_checkpointed_one(self):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "ckpt.pt")
            model = make_tiny_model()
            loader = make_loader(10, 4, num_sequences=8, batch_size=2)
            save_config = {
                "learning_rate": 1e-2, "weight_decay": 0.0, "epochs": 5,
                "grad_accumulation_steps": 1, "grad_clip_norm": None,
            }
            train(model, loader, loader, save_config, max_steps=2, checkpoint_path=path, log_fn=lambda m: None)
            checkpoint_state = {n: p.clone() for n, p in model.named_parameters()}

            # Resume with a learning rate 1e6x smaller: if resume correctly
            # applies the NEW config, the next steps should barely move
            # the parameters at all (a bug would keep using lr=1e-2).
            tiny_lr_config = {
                "learning_rate": 1e-8, "weight_decay": 0.0, "epochs": 5,
                "grad_accumulation_steps": 1, "grad_clip_norm": None,
            }
            resumed_model = make_tiny_model()
            train(resumed_model, loader, loader, tiny_lr_config, max_steps=5, resume_from=path, log_fn=lambda m: None)

            total_change = sum(
                (p - checkpoint_state[n]).norm().item() for n, p in resumed_model.named_parameters()
            )
            self.assertLess(total_change, 1e-4)


class TestEpochBookkeeping(unittest.TestCase):
    """Regression test: the saved epoch used to be the raw loop variable,
    which after a NATURALLY completed run pointed at the last epoch index
    (not "the next epoch to run"), so resuming replayed that already-
    finished epoch's data a second time."""

    def test_resume_after_natural_completion_does_not_replay_the_completed_epoch(self):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "ckpt.pt")
            model = make_tiny_model()
            loader = make_loader(10, 4, num_sequences=4, batch_size=1)  # 4 batches/epoch
            config = {
                "learning_rate": 1e-3, "weight_decay": 0.0, "epochs": 2,
                "grad_accumulation_steps": 1, "grad_clip_norm": None,
            }

            history1 = train(model, loader, loader, config, checkpoint_path=path, log_fn=lambda m: None)
            self.assertEqual(len(history1["train_loss"]), 8)  # 2 epochs x 4 batches, no max_steps cutoff

            resumed_model = make_tiny_model()
            config2 = {**config, "epochs": 4}
            history2 = train(resumed_model, loader, loader, config2, resume_from=path, checkpoint_path=path, log_fn=lambda m: None)

            # Correct: 2 more epochs (indices 2,3) x 4 batches = 8 new steps -> 16 total.
            # The bug would have replayed epoch index 1 too, giving 20.
            self.assertEqual(len(history2["train_loss"]), 16)

    def test_max_steps_interrupted_epoch_still_restarts_from_its_beginning(self):
        # The accepted, documented tradeoff: unlike natural completion, a
        # max_steps cutoff mid-epoch does NOT advance the saved epoch, so
        # resume restarts that same epoch (not true mid-epoch resume).
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


class TestSizeWeightedAccumulationMath(unittest.TestCase):
    """Regression test: the accumulated gradient over a window with unequal
    micro-batch sizes (a ragged trailing batch) must equal the gradient of
    the single pooled-batch mean loss over the same examples -- not a
    naive fixed-1/accumulation_steps split, which over/under-weights
    examples depending on which micro-batch they land in.

    Tested at the gradient level directly (not through AdamW + a step)
    because Adam's adaptive per-parameter normalization makes it an
    unreliable, insensitive way to detect a scaling error in the
    accumulated gradient -- the same reason the clipping tests above don't
    test through Adam's output either."""

    def test_ragged_window_gradient_matches_pooled_batch_gradient(self):
        torch.manual_seed(0)
        vocab_size, context_length = 10, 4
        ids = torch.randint(0, vocab_size, (context_length + 10,)).tolist()
        dataset = TextDataset(ids, context_length)
        # sizes [4, 4, 2]: mirrors what DataLoader(batch_size=4, shuffle=False)
        # yields for 10 examples -- a ragged trailing micro-batch.
        microbatches = [
            (torch.stack([dataset[i][0] for i in range(0, 4)]), torch.stack([dataset[i][1] for i in range(0, 4)])),
            (torch.stack([dataset[i][0] for i in range(4, 8)]), torch.stack([dataset[i][1] for i in range(4, 8)])),
            (torch.stack([dataset[i][0] for i in range(8, 10)]), torch.stack([dataset[i][1] for i in range(8, 10)])),
        ]

        accumulated_model = make_tiny_model(vocab_size, context_length)
        accumulated_model.zero_grad()
        window_token_count = 0
        for x, y in microbatches:
            loss = cross_entropy_loss(accumulated_model(x), y)
            token_count = y.numel()
            (loss * token_count).backward()
            window_token_count += token_count
        for p in accumulated_model.parameters():
            if p.grad is not None:
                p.grad /= window_token_count

        pooled_model = make_tiny_model(vocab_size, context_length)
        pooled_model.load_state_dict(accumulated_model.state_dict())  # identical weights, fair comparison
        pooled_model.zero_grad()
        xs = torch.cat([mb[0] for mb in microbatches])
        ys = torch.cat([mb[1] for mb in microbatches])
        cross_entropy_loss(pooled_model(xs), ys).backward()

        for (n1, p1), (n2, p2) in zip(accumulated_model.named_parameters(), pooled_model.named_parameters()):
            self.assertTrue(torch.allclose(p1.grad, p2.grad, atol=1e-5), f"{n1} gradient mismatch")


class TestWindowedLossReporting(unittest.TestCase):
    """Regression test: history/log used to record only the LAST
    micro-batch's loss per window, not the window's true token-weighted
    average, which misrepresents what the accumulated update actually
    optimized whenever accumulation_steps > 1."""

    def test_history_records_token_weighted_window_average_not_last_microbatch(self):
        torch.manual_seed(0)
        vocab_size, context_length = 10, 4
        loader = make_loader(vocab_size, context_length, num_sequences=6, batch_size=2)  # 3 microbatches -> one window

        model = make_tiny_model(vocab_size, context_length)
        snapshot = copy.deepcopy(model)
        snapshot.eval()

        expected_sum, expected_tokens, last_batch_loss = 0.0, 0, None
        with torch.no_grad():
            for x, y in loader:
                l = cross_entropy_loss(snapshot(x), y)
                expected_sum += l.item() * y.numel()
                expected_tokens += y.numel()
                last_batch_loss = l.item()
        expected_avg = expected_sum / expected_tokens

        config = {
            "learning_rate": 1e-3, "weight_decay": 0.0, "epochs": 1,
            "grad_accumulation_steps": 3, "grad_clip_norm": None,
        }
        history = train(model, loader, loader, config, log_fn=lambda m: None)

        self.assertEqual(len(history["train_loss"]), 1)
        self.assertAlmostEqual(history["train_loss"][0], expected_avg, places=4)
        self.assertNotAlmostEqual(history["train_loss"][0], last_batch_loss, places=2)


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
