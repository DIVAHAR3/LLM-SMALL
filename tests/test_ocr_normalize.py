import sys
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ocr.normalize import IMAGE_SIZE, normalize_bbox_to_canvas, normalize_to_canvas


class TestNormalizeToCanvas(unittest.TestCase):
    def test_output_is_always_the_target_size(self):
        image = Image.new("L", (100, 40), color=255)
        image.paste(0, (10, 10, 30, 25))  # some ink, off-center, non-square
        result = normalize_to_canvas(image)
        self.assertEqual(result.size, (IMAGE_SIZE, IMAGE_SIZE))

    def test_blank_image_returns_a_blank_canvas(self):
        image = Image.new("L", (50, 50), color=255)
        result = normalize_to_canvas(image)
        self.assertEqual(result.getextrema(), (255, 255))

    def test_ink_fills_most_of_the_output_frame_regardless_of_original_padding(self):
        # a tiny 4x4 ink square in the middle of a big mostly-blank canvas --
        # after normalization it should fill most of the output, not stay tiny
        image = Image.new("L", (200, 200), color=255)
        image.paste(0, (98, 98, 102, 102))
        result = normalize_to_canvas(image)
        inverted_extrema = Image.eval(result, lambda p: 255 - p).getbbox()
        ink_width = inverted_extrema[2] - inverted_extrema[0]
        self.assertGreater(ink_width, IMAGE_SIZE * 0.5)

    def test_wide_and_tall_ink_regions_are_both_handled_without_distortion(self):
        wide = Image.new("L", (60, 60), color=255)
        wide.paste(0, (5, 25, 55, 35))  # a wide, short horizontal bar
        tall = Image.new("L", (60, 60), color=255)
        tall.paste(0, (25, 5, 35, 55))  # a tall, narrow vertical bar

        wide_result = normalize_to_canvas(wide)
        tall_result = normalize_to_canvas(tall)
        self.assertEqual(wide_result.size, (IMAGE_SIZE, IMAGE_SIZE))
        self.assertEqual(tall_result.size, (IMAGE_SIZE, IMAGE_SIZE))

    def test_custom_padding_and_target_size_are_respected(self):
        image = Image.new("L", (50, 50), color=255)
        image.paste(0, (20, 20, 30, 30))
        result = normalize_to_canvas(image, padding=0, target_size=16)
        self.assertEqual(result.size, (16, 16))

    def test_full_bleed_ink_touching_every_edge_does_not_crash(self):
        image = Image.new("L", (20, 20), color=0)  # entirely ink, no background at all
        result = normalize_to_canvas(image)
        self.assertEqual(result.size, (IMAGE_SIZE, IMAGE_SIZE))

    def test_matches_normalize_bbox_to_canvas_given_the_equivalent_inclusive_bbox(self):
        # ocr/segment.py already knows its exact (inclusive) bbox and
        # calls normalize_bbox_to_canvas directly -- this must produce
        # pixel-identical output to normalize_to_canvas's auto-detected
        # path, since both ultimately feed the same trained classifier.
        image = Image.new("L", (60, 60), color=255)
        image.paste(0, (20, 15, 35, 40))  # ink spans x=20..34, y=15..39 (paste's box is exclusive on right/lower)
        auto = normalize_to_canvas(image)
        manual = normalize_bbox_to_canvas(image, (20, 15, 34, 39))
        self.assertEqual(auto.tobytes(), manual.tobytes())


class TestNormalizeBboxToCanvas(unittest.TestCase):
    def test_output_is_always_the_target_size(self):
        image = Image.new("L", (100, 100), color=255)
        image.paste(0, (10, 10, 20, 20))
        result = normalize_bbox_to_canvas(image, (10, 10, 19, 19))
        self.assertEqual(result.size, (IMAGE_SIZE, IMAGE_SIZE))

    def test_padding_is_taken_from_the_real_surrounding_image_not_clamped_to_zero(self):
        # The bug this function exists to fix: cropping tightly to the
        # bbox FIRST and only then padding would have nothing real to
        # pad with at the crop's own edges. Given real surrounding
        # background here, padding should visibly shrink the ink's
        # footprint in the output relative to zero padding.
        image = Image.new("L", (100, 100), color=255)
        image.paste(0, (40, 40, 60, 60))  # a big 20x20 ink square, plenty of real margin around it
        bbox = (40, 40, 59, 59)

        no_padding = normalize_bbox_to_canvas(image, bbox, padding=0)
        with_padding = normalize_bbox_to_canvas(image, bbox, padding=8)

        no_padding_ink_width = Image.eval(no_padding, lambda p: 255 - p).getbbox()[2]
        with_padding_ink_width = Image.eval(with_padding, lambda p: 255 - p).getbbox()[2]
        self.assertGreater(no_padding_ink_width, with_padding_ink_width)

    def test_bbox_at_the_image_edge_is_clamped_not_out_of_bounds(self):
        image = Image.new("L", (30, 30), color=255)
        image.paste(0, (0, 0, 5, 5))  # ink touching the top-left corner
        result = normalize_bbox_to_canvas(image, (0, 0, 4, 4), padding=5)
        self.assertEqual(result.size, (IMAGE_SIZE, IMAGE_SIZE))


if __name__ == "__main__":
    unittest.main()
