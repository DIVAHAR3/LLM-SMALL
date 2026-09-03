import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ocr.synthetic_data import CHARACTERS, IMAGE_SIZE, _available_fonts, generate_dataset, render_character


@unittest.skipUnless(_available_fonts(), "no usable system fonts found -- synthetic OCR data needs at least one")
class TestGenerateDataset(unittest.TestCase):
    def test_produces_the_requested_number_of_samples_per_character(self):
        examples = generate_dataset(samples_per_character=3, seed=0)
        self.assertEqual(len(examples), len(CHARACTERS) * 3)

    def test_every_character_in_the_alphabet_is_represented(self):
        examples = generate_dataset(samples_per_character=2, seed=0)
        labels = {label for _, label in examples}
        self.assertEqual(labels, set(CHARACTERS))

    def test_each_character_appears_exactly_samples_per_character_times(self):
        examples = generate_dataset(samples_per_character=4, seed=0)
        counts = {}
        for _, label in examples:
            counts[label] = counts.get(label, 0) + 1
        self.assertTrue(all(count == 4 for count in counts.values()))

    def test_images_are_the_expected_size_and_mode(self):
        examples = generate_dataset(samples_per_character=1, seed=0)
        image, _ = examples[0]
        self.assertEqual(image.size, (IMAGE_SIZE, IMAGE_SIZE))
        self.assertEqual(image.mode, "L")

    def test_is_deterministic_given_the_same_seed(self):
        first = generate_dataset(samples_per_character=2, seed=42)
        second = generate_dataset(samples_per_character=2, seed=42)
        self.assertEqual([img.tobytes() for img, _ in first], [img.tobytes() for img, _ in second])
        self.assertEqual([label for _, label in first], [label for _, label in second])

    def test_different_seeds_produce_different_renderings(self):
        first = generate_dataset(samples_per_character=2, seed=1)
        second = generate_dataset(samples_per_character=2, seed=2)
        first_bytes = [img.tobytes() for img, _ in first]
        second_bytes = [img.tobytes() for img, _ in second]
        self.assertNotEqual(first_bytes, second_bytes)

    def test_rejects_zero_or_fewer_samples_gracefully(self):
        # not an error case worth raising on -- just produces an empty dataset
        examples = generate_dataset(samples_per_character=0, seed=0)
        self.assertEqual(examples, [])


@unittest.skipUnless(_available_fonts(), "no usable system fonts found -- synthetic OCR data needs at least one")
class TestRenderCharacter(unittest.TestCase):
    def test_rendered_character_is_not_a_blank_canvas(self):
        font_path = _available_fonts()[0]
        rng = random.Random(0)
        image = render_character("A", font_path, 20, rng)
        self.assertNotEqual(image.getextrema(), (255, 255))  # something darker than pure white was drawn

    def test_same_rng_state_produces_identical_images(self):
        font_path = _available_fonts()[0]
        image_a = render_character("A", font_path, 20, random.Random(7))
        image_b = render_character("A", font_path, 20, random.Random(7))
        self.assertEqual(image_a.tobytes(), image_b.tobytes())

    def test_different_characters_produce_different_images(self):
        font_path = _available_fonts()[0]
        image_a = render_character("A", font_path, 20, random.Random(0))
        image_b = render_character("Z", font_path, 20, random.Random(0))
        self.assertNotEqual(image_a.tobytes(), image_b.tobytes())


if __name__ == "__main__":
    unittest.main()
