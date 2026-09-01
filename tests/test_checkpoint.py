import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.gpt import GPTModel
from training.checkpoint import load_checkpoint, save_checkpoint


def make_tiny_model():
    return GPTModel(
        vocab_size=15, context_length=10, embedding_dim=8,
        num_layers=2, num_heads=2, ffn_hidden_dim=16, dropout=0.0,
    )


class TestCheckpointRoundTrip(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = make_tiny_model()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-2)

        # Take a couple of real optimizer steps so Adam actually has
        # non-trivial running-average state to verify gets restored.
        x = torch.randint(0, 15, (2, 10))
        y = torch.randint(0, 15, (2, 10))
        for _ in range(2):
            logits = self.model(x)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 15), y.reshape(-1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmpdir.name) / "ckpt.pt")
        self.metrics = {"train_loss": [1.0, 0.8], "val_loss": [1.1]}
        save_checkpoint(self.path, self.model, self.optimizer, epoch=3, step=7, config={"lr": 1e-2}, metrics=self.metrics)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_epoch_step_config_metrics_round_trip(self):
        fresh_model = make_tiny_model()
        checkpoint = load_checkpoint(self.path, fresh_model)
        self.assertEqual(checkpoint["epoch"], 3)
        self.assertEqual(checkpoint["step"], 7)
        self.assertEqual(checkpoint["config"], {"lr": 1e-2})
        self.assertEqual(checkpoint["metrics"], self.metrics)

    def test_model_weights_are_bit_identical_after_reload(self):
        # A freshly-initialized model has different random weights; loading
        # must fully overwrite them, not merge/leave any untouched.
        fresh_model = make_tiny_model()
        before_reload = {n: p.clone() for n, p in fresh_model.named_parameters()}
        load_checkpoint(self.path, fresh_model)
        for name, param in fresh_model.named_parameters():
            self.assertFalse(torch.equal(before_reload[name], param), f"{name} unchanged by load")
        for (n1, p1), (n2, p2) in zip(self.model.named_parameters(), fresh_model.named_parameters()):
            self.assertEqual(n1, n2)
            self.assertTrue(torch.equal(p1, p2), f"{n1} does not match saved model exactly")

    def test_reload_produces_identical_forward_pass_output(self):
        fresh_model = make_tiny_model()
        load_checkpoint(self.path, fresh_model)
        fresh_model.eval()
        self.model.eval()

        x = torch.randint(0, 15, (1, 10))
        with torch.no_grad():
            original_out = self.model(x)
            reloaded_out = fresh_model(x)
        self.assertTrue(torch.equal(original_out, reloaded_out))

    def test_optimizer_state_is_restored(self):
        fresh_model = make_tiny_model()
        fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=1e-2)
        load_checkpoint(self.path, fresh_model, fresh_optimizer)

        original_state = self.optimizer.state_dict()["state"]
        reloaded_state = fresh_optimizer.state_dict()["state"]
        self.assertEqual(set(original_state.keys()), set(reloaded_state.keys()))
        for key in original_state:
            self.assertTrue(torch.equal(original_state[key]["exp_avg"], reloaded_state[key]["exp_avg"]))
            self.assertTrue(torch.equal(original_state[key]["exp_avg_sq"], reloaded_state[key]["exp_avg_sq"]))

    def test_resumed_training_step_matches_continuing_without_reload(self):
        """The real test of resume correctness: one more optimizer step on
        the reloaded model+optimizer must match one more step on the
        original, unbroken model+optimizer, given the same batch."""
        x = torch.randint(0, 15, (2, 10))
        y = torch.randint(0, 15, (2, 10))

        fresh_model = make_tiny_model()
        fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=1e-2)
        load_checkpoint(self.path, fresh_model, fresh_optimizer)

        for model, optimizer in [(self.model, self.optimizer), (fresh_model, fresh_optimizer)]:
            model.train()
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 15), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for (n1, p1), (n2, p2) in zip(self.model.named_parameters(), fresh_model.named_parameters()):
            self.assertTrue(torch.equal(p1, p2), f"{n1} diverged after one resumed step")


if __name__ == "__main__":
    unittest.main()
