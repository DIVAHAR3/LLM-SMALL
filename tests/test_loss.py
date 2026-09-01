import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.gpt import GPTModel
from tokenizer.char_tokenizer import CharTokenizer
from training.dataset import TextDataset
from training.loss import cross_entropy_loss


class TestCrossEntropyLoss(unittest.TestCase):
    def test_matches_pytorch_reference_implementation(self):
        torch.manual_seed(0)
        logits = torch.randn(3, 5, 10)
        targets = torch.randint(0, 10, (3, 5))
        ours = cross_entropy_loss(logits, targets)
        reference = F.cross_entropy(logits.reshape(-1, 10), targets.reshape(-1))
        self.assertTrue(torch.allclose(ours, reference, atol=1e-6))

    def test_uniform_logits_give_log_vocab_size(self):
        # softmax(0s) is uniform 1/V -> -log(1/V) == log(V), exactly the
        # "as confused as random guessing" baseline described above.
        vocab_size = 7
        logits = torch.zeros(2, 4, vocab_size)
        targets = torch.randint(0, vocab_size, (2, 4))
        loss = cross_entropy_loss(logits, targets)
        self.assertAlmostEqual(loss.item(), math.log(vocab_size), places=5)

    def test_confident_correct_prediction_gives_near_zero_loss(self):
        vocab_size = 5
        logits = torch.full((1, 1, vocab_size), -100.0)
        logits[0, 0, 2] = 100.0  # overwhelmingly favors class 2
        targets = torch.tensor([[2]])
        loss = cross_entropy_loss(logits, targets)
        self.assertLess(loss.item(), 1e-3)

    def test_confident_wrong_prediction_gives_large_loss(self):
        vocab_size = 5
        logits = torch.full((1, 1, vocab_size), -100.0)
        logits[0, 0, 2] = 100.0  # confidently predicts class 2
        targets = torch.tensor([[0]])  # but the true class is 0
        loss = cross_entropy_loss(logits, targets)
        self.assertGreater(loss.item(), 50.0)

    def test_gradients_flow_to_logits(self):
        logits = torch.randn(2, 3, 6, requires_grad=True)
        targets = torch.randint(0, 6, (2, 3))
        loss = cross_entropy_loss(logits, targets)
        loss.backward()
        self.assertIsNotNone(logits.grad)


class TestLossDecreasesOnTinyDataset(unittest.TestCase):
    def test_loss_drops_substantially_while_overfitting_a_real_batch(self):
        """The phase's stop condition: wire tokenizer -> dataset -> model ->
        loss -> optimizer together and confirm loss actually decreases on
        real next-token-prediction targets, not just synthetic numbers."""
        torch.manual_seed(0)
        text = "the quick brown fox jumps over the lazy dog. " * 4
        tokenizer = CharTokenizer.from_text(text)
        ids = tokenizer.encode(text)

        context_length = 16
        dataset = TextDataset(ids, context_length)
        loader = DataLoader(dataset, batch_size=4, shuffle=True, generator=torch.Generator().manual_seed(0))
        x, y = next(iter(loader))

        model = GPTModel(
            tokenizer.vocab_size, context_length, embedding_dim=16,
            num_layers=2, num_heads=2, ffn_hidden_dim=32, dropout=0.0,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

        losses = []
        for _ in range(80):
            logits = model(x)
            loss = cross_entropy_loss(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        self.assertLess(losses[-1], losses[0] * 0.5)


if __name__ == "__main__":
    unittest.main()
