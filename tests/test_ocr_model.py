import json
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ocr.model import CharacterCNN, validate_ocr_model_config


class TestCharacterCNN(unittest.TestCase):
    def setUp(self):
        self.image_size = 16
        self.num_classes = 10
        self.model = CharacterCNN(
            self.image_size, self.num_classes, conv1_channels=4, conv2_channels=8, hidden_dim=32, dropout=0.0,
        )
        self.model.eval()

    def test_forward_pass_output_shape(self):
        images = torch.randn(5, 1, self.image_size, self.image_size)
        logits = self.model(images)
        self.assertEqual(logits.shape, (5, self.num_classes))

    def test_forward_pass_output_is_finite(self):
        images = torch.randn(3, 1, self.image_size, self.image_size)
        logits = self.model(images)
        self.assertTrue(torch.isfinite(logits).all())

    def test_gradients_flow_to_every_component(self):
        model = CharacterCNN(
            self.image_size, self.num_classes, conv1_channels=4, conv2_channels=8, hidden_dim=32, dropout=0.0,
        )
        images = torch.randn(2, 1, self.image_size, self.image_size)
        loss = model(images).sum()
        loss.backward()
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"{name} got no gradient")

    def test_num_parameters_matches_manual_sum(self):
        manual_total = sum(p.numel() for p in self.model.parameters())
        self.assertEqual(self.model.num_parameters(), manual_total)

    def test_summary_contains_component_breakdown(self):
        text = self.model.summary()
        for expected in ["conv1", "conv2", "fc1", "fc2", "TOTAL"]:
            self.assertIn(expected, text)

    def test_from_config_builds_from_ocr_model_config_json(self):
        config = json.loads((ROOT / "configs" / "ocr_model_config.json").read_text(encoding="utf-8"))
        model = CharacterCNN.from_config(config)
        images = torch.randn(2, 1, config["image_size"], config["image_size"])
        logits = model(images)
        self.assertEqual(logits.shape, (2, config["num_classes"]))

    def test_different_image_sizes_produce_correctly_shaped_output(self):
        model = CharacterCNN(32, 5, conv1_channels=4, conv2_channels=8, hidden_dim=16, dropout=0.0)
        images = torch.randn(1, 1, 32, 32)
        logits = model(images)
        self.assertEqual(logits.shape, (1, 5))


class TestValidateOcrModelConfig(unittest.TestCase):
    def setUp(self):
        self.valid_config = {
            "image_size": 16, "num_classes": 10, "conv1_channels": 4,
            "conv2_channels": 8, "hidden_dim": 32, "dropout": 0.2,
        }

    def test_valid_config_raises_nothing(self):
        validate_ocr_model_config(self.valid_config)  # must not raise

    def test_the_real_project_config_is_valid(self):
        config = json.loads((ROOT / "configs" / "ocr_model_config.json").read_text(encoding="utf-8"))
        validate_ocr_model_config(config)  # must not raise

    def test_rejects_missing_required_key(self):
        del self.valid_config["hidden_dim"]
        with self.assertRaises(ValueError):
            validate_ocr_model_config(self.valid_config)

    def test_rejects_non_positive_value(self):
        self.valid_config["conv1_channels"] = 0
        with self.assertRaises(ValueError):
            validate_ocr_model_config(self.valid_config)

    def test_rejects_non_integer_value(self):
        self.valid_config["hidden_dim"] = 32.5
        with self.assertRaises(ValueError):
            validate_ocr_model_config(self.valid_config)

    def test_rejects_dropout_out_of_range(self):
        self.valid_config["dropout"] = 1.5
        with self.assertRaises(ValueError):
            validate_ocr_model_config(self.valid_config)

    def test_rejects_image_size_not_divisible_by_4(self):
        self.valid_config["image_size"] = 15
        with self.assertRaises(ValueError):
            validate_ocr_model_config(self.valid_config)

    def test_from_config_rejects_invalid_config_before_building_anything(self):
        bad_config = dict(self.valid_config, image_size=15)
        with self.assertRaises(ValueError):
            CharacterCNN.from_config(bad_config)


if __name__ == "__main__":
    unittest.main()
