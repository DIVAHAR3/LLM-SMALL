import random
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ocr.dataset import CharacterDataset
from ocr.model import CharacterCNN
from ocr.synthetic_data import _available_fonts, render_character
from ocr.train import evaluate, train
from training.checkpoint import load_checkpoint


def make_tiny_learnable_dataset(characters="ABC", samples_per_character=12, seed=0):
    """A handful of characters, each rendered many times with jitter --
    same generation mechanism as the real dataset, just far fewer
    classes, so the toy model below can overfit it in a handful of
    epochs (same "start tiny, overfit a toy dataset end-to-end"
    approach this project used for the GPT itself)."""
    fonts = _available_fonts()
    rng = random.Random(seed)
    examples = []
    for char in characters:
        for _ in range(samples_per_character):
            examples.append((render_character(char, fonts[0], 20, rng), char))
    return examples


@unittest.skipUnless(_available_fonts(), "no usable system fonts found -- OCR training needs at least one")
class TestTrain(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.characters = "ABC"
        examples = make_tiny_learnable_dataset(self.characters, samples_per_character=16, seed=1)
        split = int(len(examples) * 0.8)
        self.train_loader = DataLoader(CharacterDataset(examples[:split]), batch_size=8, shuffle=True)
        self.val_loader = DataLoader(CharacterDataset(examples[split:]), batch_size=8, shuffle=False)
        self.model = CharacterCNN(image_size=28, num_classes=len(self.characters), conv1_channels=4, conv2_channels=8, hidden_dim=16, dropout=0.0)

    def test_loss_visibly_decreases_over_training(self):
        history = train(self.model, self.train_loader, self.val_loader, learning_rate=1e-2, epochs=8, log_fn=lambda msg: None)
        self.assertLess(history["train_loss"][-1], history["train_loss"][0])

    def test_model_learns_to_overfit_this_tiny_dataset(self):
        history = train(self.model, self.train_loader, self.val_loader, learning_rate=1e-2, epochs=15, log_fn=lambda msg: None)
        self.assertGreaterEqual(history["val_accuracy"][-1], 0.8)

    def test_saves_a_self_describing_checkpoint_when_a_path_is_given(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = str(Path(tmpdir) / "ocr_test.pt")
            train(self.model, self.train_loader, self.val_loader, epochs=1, checkpoint_path=checkpoint_path, log_fn=lambda msg: None)

            fresh_model = CharacterCNN(image_size=28, num_classes=len(self.characters), conv1_channels=4, conv2_channels=8, hidden_dim=16, dropout=0.0)
            checkpoint = load_checkpoint(checkpoint_path, fresh_model)
            self.assertEqual(checkpoint["model_config"], self.model.config)


@unittest.skipUnless(_available_fonts(), "no usable system fonts found -- OCR training needs at least one")
class TestEvaluate(unittest.TestCase):
    def test_returns_loss_and_accuracy_keys(self):
        torch.manual_seed(0)
        examples = make_tiny_learnable_dataset("AB", samples_per_character=4, seed=2)
        loader = DataLoader(CharacterDataset(examples), batch_size=4, shuffle=False)
        model = CharacterCNN(image_size=28, num_classes=2, conv1_channels=4, conv2_channels=8, hidden_dim=16, dropout=0.0)

        result = evaluate(model, loader)
        self.assertIn("loss", result)
        self.assertIn("accuracy", result)
        self.assertGreaterEqual(result["accuracy"], 0.0)
        self.assertLessEqual(result["accuracy"], 1.0)

    def test_leaves_model_train_mode_unchanged_after_returning(self):
        examples = make_tiny_learnable_dataset("AB", samples_per_character=4, seed=2)
        loader = DataLoader(CharacterDataset(examples), batch_size=4, shuffle=False)
        model = CharacterCNN(image_size=28, num_classes=2, conv1_channels=4, conv2_channels=8, hidden_dim=16, dropout=0.0)

        model.train()
        evaluate(model, loader)
        self.assertTrue(model.training)

        model.eval()
        evaluate(model, loader)
        self.assertFalse(model.training)


if __name__ == "__main__":
    unittest.main()
