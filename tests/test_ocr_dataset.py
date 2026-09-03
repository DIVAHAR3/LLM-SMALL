import sys
import unittest
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ocr.dataset import CHAR_TO_INDEX, INDEX_TO_CHAR, CharacterDataset, image_to_tensor
from ocr.synthetic_data import CHARACTERS


class TestImageToTensor(unittest.TestCase):
    def test_shape_and_dtype(self):
        image = Image.new("L", (28, 28), color=128)
        tensor = image_to_tensor(image)
        self.assertEqual(tensor.shape, (1, 28, 28))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_white_image_maps_to_all_ones(self):
        image = Image.new("L", (10, 10), color=255)
        tensor = image_to_tensor(image)
        self.assertTrue(torch.allclose(tensor, torch.ones_like(tensor)))

    def test_black_image_maps_to_all_zeros(self):
        image = Image.new("L", (10, 10), color=0)
        tensor = image_to_tensor(image)
        self.assertTrue(torch.allclose(tensor, torch.zeros_like(tensor)))

    def test_non_square_image_preserves_width_and_height_order(self):
        image = Image.new("L", (12, 6), color=0)  # width=12, height=6
        tensor = image_to_tensor(image)
        self.assertEqual(tensor.shape, (1, 6, 12))  # (C, H, W)

    def test_rgb_image_is_converted_to_grayscale_first(self):
        image = Image.new("RGB", (5, 5), color=(255, 255, 255))
        tensor = image_to_tensor(image)
        self.assertEqual(tensor.shape, (1, 5, 5))
        self.assertTrue(torch.allclose(tensor, torch.ones_like(tensor)))


class TestCharToIndexMapping(unittest.TestCase):
    def test_every_character_has_a_unique_index(self):
        self.assertEqual(len(CHAR_TO_INDEX), len(CHARACTERS))
        self.assertEqual(len(set(CHAR_TO_INDEX.values())), len(CHARACTERS))

    def test_index_to_char_is_the_exact_inverse(self):
        for char, index in CHAR_TO_INDEX.items():
            self.assertEqual(INDEX_TO_CHAR[index], char)

    def test_indices_are_contiguous_from_zero(self):
        self.assertEqual(sorted(CHAR_TO_INDEX.values()), list(range(len(CHARACTERS))))


class TestCharacterDataset(unittest.TestCase):
    def setUp(self):
        self.examples = [
            (Image.new("L", (28, 28), color=255), "A"),
            (Image.new("L", (28, 28), color=0), "b"),
            (Image.new("L", (28, 28), color=128), "5"),
        ]
        self.dataset = CharacterDataset(self.examples)

    def test_len_matches_number_of_examples(self):
        self.assertEqual(len(self.dataset), 3)

    def test_getitem_returns_tensor_and_correct_label_index(self):
        tensor, label_index = self.dataset[0]
        self.assertEqual(tensor.shape, (1, 28, 28))
        self.assertEqual(label_index, CHAR_TO_INDEX["A"])

    def test_labels_round_trip_through_index_to_char(self):
        for i, (_, char) in enumerate(self.examples):
            _, label_index = self.dataset[i]
            self.assertEqual(INDEX_TO_CHAR[label_index], char)

    def test_works_with_a_real_dataloader(self):
        loader = DataLoader(self.dataset, batch_size=2, shuffle=False)
        images, labels = next(iter(loader))
        self.assertEqual(images.shape, (2, 1, 28, 28))
        self.assertEqual(labels.shape, (2,))


if __name__ == "__main__":
    unittest.main()
