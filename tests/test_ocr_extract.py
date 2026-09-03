import random
import sys
import unittest
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ocr.dataset import CHAR_TO_INDEX
from ocr.extract import _line_to_text, extract_text
from ocr.synthetic_data import CHARACTERS, IMAGE_SIZE, _available_fonts, render_character


class _FixedPredictionModel:
    """Test double: always predicts the same character regardless of
    input, so extract_text's SPACE-INSERTION logic (the actual thing
    this module adds beyond segmentation+classification) can be tested
    in isolation from real classification accuracy."""

    def __init__(self, char):
        self.index = CHAR_TO_INDEX[char]

    def __call__(self, tensor):
        logits = torch.full((tensor.shape[0], len(CHARACTERS)), -100.0)
        logits[:, self.index] = 100.0
        return logits


def make_line(gaps, box_width=10, box_height=20, y=0):
    """A fake (crop, box) line with EXACT, controlled gaps between
    consecutive boxes -- crop content doesn't matter since
    _FixedPredictionModel ignores it entirely."""
    blank_crop = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=255)
    boxes = []
    x = 0
    for i, gap in enumerate([0] + list(gaps)):
        x += gap if i > 0 else 0
        boxes.append((x, y, x + box_width - 1, y + box_height - 1))
        x += box_width
    return [(blank_crop, box) for box in boxes]


class TestLineToText(unittest.TestCase):
    def test_empty_line_returns_empty_string(self):
        self.assertEqual(_line_to_text(_FixedPredictionModel("A"), []), "")

    def test_single_character_line_has_no_spaces(self):
        line = make_line([])
        self.assertEqual(_line_to_text(_FixedPredictionModel("A"), line), "A")

    def test_uniform_small_gaps_produce_no_spaces(self):
        line = make_line([3, 3, 3, 3])
        self.assertEqual(_line_to_text(_FixedPredictionModel("A"), line), "AAAAA")

    def test_one_outlier_gap_becomes_a_single_space(self):
        line = make_line([2, 2, 20, 2, 2])
        self.assertEqual(_line_to_text(_FixedPredictionModel("A"), line), "AAA AAA")

    def test_multiple_outlier_gaps_each_become_a_space(self):
        line = make_line([2, 15, 2, 15, 2])
        self.assertEqual(_line_to_text(_FixedPredictionModel("A"), line), "AA AA AA")

    def test_predicted_characters_come_from_the_model_not_hardcoded(self):
        line = make_line([2, 2])
        self.assertEqual(_line_to_text(_FixedPredictionModel("Z"), line), "ZZZ")


@unittest.skipUnless(_available_fonts(), "no usable system fonts found -- OCR extraction needs at least one")
class TestExtractText(unittest.TestCase):
    def _render_row(self, word, width_chars, font_size=22):
        font_path = _available_fonts()[0]
        rng = random.Random(0)
        canvas = Image.new("L", (IMAGE_SIZE * width_chars, IMAGE_SIZE), color=255)
        for i, ch in enumerate(word):
            canvas.paste(render_character(ch, font_path, font_size, rng, jitter_px=0, max_rotation_degrees=0), (i * IMAGE_SIZE, 0))
        return canvas

    def test_two_lines_are_joined_with_a_newline(self):
        top = self._render_row("ABC", 3)
        bottom = self._render_row("DE", 2)
        canvas = Image.new("L", (IMAGE_SIZE * 3, IMAGE_SIZE * 3), color=255)
        canvas.paste(top, (0, 0))
        canvas.paste(bottom, (0, IMAGE_SIZE * 2))

        text = extract_text(canvas, _FixedPredictionModel("Z"))
        self.assertEqual(text, "ZZZ\nZZ")

    def test_returns_empty_string_for_a_blank_image(self):
        blank = Image.new("L", (50, 50), color=255)
        self.assertEqual(extract_text(blank, _FixedPredictionModel("A")), "")

    def test_uses_the_real_trained_checkpoint_end_to_end_without_crashing(self):
        # Not an accuracy assertion (documented as limited on real text,
        # see docs/OCR.md) -- just confirms the whole real pipeline
        # (segment -> normalize -> classify -> reconstruct) runs
        # end-to-end and returns a plausible-shaped string.
        from ocr.checkpoint import load_ocr_model_for_inference

        checkpoint_path = Path(__file__).resolve().parent.parent / "checkpoints" / "ocr_character_cnn.pt"
        if not checkpoint_path.exists():
            self.skipTest("no trained OCR checkpoint present -- run scripts/train_ocr.py first")

        model, _ = load_ocr_model_for_inference(str(checkpoint_path))
        canvas = self._render_row("HELLO", 5)
        text = extract_text(canvas, model)
        self.assertEqual(len(text), 5)
        self.assertTrue(text.isalpha())


if __name__ == "__main__":
    unittest.main()
