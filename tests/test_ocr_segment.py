import random
import sys
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ocr.segment import (
    find_connected_components,
    group_into_lines,
    group_into_reading_order,
    otsu_threshold,
    segment_characters,
    segment_characters_by_line,
)
from ocr.synthetic_data import IMAGE_SIZE, _available_fonts, render_character


class TestOtsuThreshold(unittest.TestCase):
    def test_splits_a_clearly_bimodal_histogram_between_the_two_peaks(self):
        # Between-class variance is flat across the whole gap (20..219) for
        # a perfectly symmetric two-cluster histogram like this one, since
        # ties resolve to the first max encountered -- deterministically 20.
        # Either way, what actually matters is that 20 and 220 land on
        # opposite sides of the threshold, which this confirms directly.
        histogram = [0] * 256
        histogram[20] = 500   # a dark cluster
        histogram[220] = 500  # a light cluster
        threshold = otsu_threshold(histogram)
        self.assertLessEqual(20, threshold)
        self.assertLess(threshold, 220)

    def test_handles_an_empty_histogram_without_crashing(self):
        self.assertEqual(otsu_threshold([0] * 256), 0)

    def test_handles_a_single_value_histogram(self):
        histogram = [0] * 256
        histogram[128] = 1000
        # degenerate case (no real split exists) -- must not raise
        otsu_threshold(histogram)


class TestFindConnectedComponents(unittest.TestCase):
    def test_two_separate_blobs_are_found_as_two_components(self):
        width, height = 10, 10
        mask = [False] * (width * height)
        for x, y in [(1, 1), (1, 2), (2, 1), (2, 2)]:  # 2x2 blob, top-left
            mask[y * width + x] = True
        for x, y in [(7, 7), (7, 8), (8, 7), (8, 8)]:  # 2x2 blob, bottom-right
            mask[y * width + x] = True

        boxes = find_connected_components(mask, width, height)
        self.assertEqual(len(boxes), 2)
        self.assertIn((1, 1, 2, 2), boxes)
        self.assertIn((7, 7, 8, 8), boxes)

    def test_diagonally_touching_pixels_count_as_one_component_8_connectivity(self):
        width, height = 5, 5
        mask = [False] * (width * height)
        mask[1 * width + 1] = True  # (1,1)
        mask[2 * width + 2] = True  # (2,2), diagonal neighbor of (1,1)

        boxes = find_connected_components(mask, width, height)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0], (1, 1, 2, 2))

    def test_empty_mask_finds_nothing(self):
        self.assertEqual(find_connected_components([False] * 25, 5, 5), [])


class TestGroupIntoLines(unittest.TestCase):
    def test_groups_boxes_into_separate_lines_preserving_structure(self):
        boxes = [
            (20, 0, 29, 9), (0, 0, 9, 9), (10, 0, 19, 9),  # line 1
            (0, 50, 9, 59),                                 # line 2
        ]
        lines = group_into_lines(boxes)
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0], [(0, 0, 9, 9), (10, 0, 19, 9), (20, 0, 29, 9)])
        self.assertEqual(lines[1], [(0, 50, 9, 59)])

    def test_flattening_group_into_lines_matches_group_into_reading_order(self):
        boxes = [(20, 0, 29, 9), (0, 0, 9, 9), (0, 50, 9, 59), (10, 0, 19, 9)]
        flat = [box for line in group_into_lines(boxes) for box in line]
        self.assertEqual(flat, group_into_reading_order(boxes))

    def test_empty_input_returns_empty_list(self):
        self.assertEqual(group_into_lines([]), [])


class TestGroupIntoReadingOrder(unittest.TestCase):
    def test_orders_two_lines_top_to_bottom_then_left_to_right(self):
        # line 1 (y=0-9): C at x=20, A at x=0, B at x=10
        # line 2 (y=50-59): D at x=0
        boxes = [
            (20, 0, 29, 9), (0, 0, 9, 9), (10, 0, 19, 9),
            (0, 50, 9, 59),
        ]
        ordered = group_into_reading_order(boxes)
        self.assertEqual(ordered, [(0, 0, 9, 9), (10, 0, 19, 9), (20, 0, 29, 9), (0, 50, 9, 59)])

    def test_empty_input_returns_empty_list(self):
        self.assertEqual(group_into_reading_order([]), [])

    def test_single_box_returns_itself(self):
        self.assertEqual(group_into_reading_order([(1, 2, 3, 4)]), [(1, 2, 3, 4)])


@unittest.skipUnless(_available_fonts(), "no usable system fonts found -- OCR segmentation needs at least one")
class TestSegmentCharacters(unittest.TestCase):
    def _render_word(self, word, font_size=22):
        """Pastes each character (rendered via the same generator used
        for training data) side by side onto one canvas -- simulating a
        short word in a pasted image, with known ground-truth order."""
        font_path = _available_fonts()[0]
        rng = random.Random(0)
        crops = [render_character(ch, font_path, font_size, rng, jitter_px=0, max_rotation_degrees=0) for ch in word]

        canvas = Image.new("L", (IMAGE_SIZE * len(word), IMAGE_SIZE), color=255)
        for i, crop in enumerate(crops):
            canvas.paste(crop, (i * IMAGE_SIZE, 0))
        return canvas

    def test_finds_one_region_per_character_in_reading_order(self):
        # avoid 'i'/'j' -- documented limitation (dot/stem split into two regions)
        canvas = self._render_word("HELLO")
        results = segment_characters(canvas)
        self.assertEqual(len(results), 5)

        # regions should appear left-to-right in the same order they were pasted
        xmins = [box[0] for _, box in results]
        self.assertEqual(xmins, sorted(xmins))

    def test_each_crop_is_the_expected_size(self):
        canvas = self._render_word("AB")
        results = segment_characters(canvas)
        for crop, _ in results:
            self.assertEqual(crop.size, (IMAGE_SIZE, IMAGE_SIZE))

    def test_works_on_light_text_on_dark_background(self):
        # inverted polarity: white ink on a black canvas
        canvas = Image.new("L", (IMAGE_SIZE * 2, IMAGE_SIZE), color=0)
        font_path = _available_fonts()[0]
        rng = random.Random(0)
        for i, ch in enumerate("AB"):
            glyph = render_character(ch, font_path, 22, rng, jitter_px=0, max_rotation_degrees=0)
            inverted = Image.eval(glyph, lambda p: 255 - p)
            canvas.paste(inverted, (i * IMAGE_SIZE, 0))

        results = segment_characters(canvas)
        self.assertEqual(len(results), 2)
        # output crops are normalized to dark-ink-on-light regardless of
        # the source image's polarity -- background corners should be light
        for crop, _ in results:
            corner_pixels = [crop.getpixel((0, 0)), crop.getpixel((IMAGE_SIZE - 1, 0))]
            self.assertTrue(all(p > 128 for p in corner_pixels))

    def test_blank_image_finds_nothing(self):
        blank = Image.new("L", (50, 50), color=255)
        self.assertEqual(segment_characters(blank), [])

    def test_segment_characters_by_line_flattens_to_the_same_result_as_segment_characters(self):
        canvas = self._render_word("HELLO")
        by_line = segment_characters_by_line(canvas)
        flat = [item for line in by_line for item in line]
        self.assertEqual([box for _, box in flat], [box for _, box in segment_characters(canvas)])

    def test_segment_characters_by_line_separates_two_stacked_words(self):
        font_path = _available_fonts()[0]
        rng = random.Random(0)
        top = Image.new("L", (IMAGE_SIZE * 2, IMAGE_SIZE), color=255)
        for i, ch in enumerate("AB"):
            top.paste(render_character(ch, font_path, 22, rng, jitter_px=0, max_rotation_degrees=0), (i * IMAGE_SIZE, 0))
        bottom = Image.new("L", (IMAGE_SIZE * 2, IMAGE_SIZE), color=255)
        for i, ch in enumerate("CD"):
            bottom.paste(render_character(ch, font_path, 22, rng, jitter_px=0, max_rotation_degrees=0), (i * IMAGE_SIZE, 0))

        canvas = Image.new("L", (IMAGE_SIZE * 2, IMAGE_SIZE * 3), color=255)
        canvas.paste(top, (0, 0))
        canvas.paste(bottom, (0, IMAGE_SIZE * 2))  # a full blank row of separation between the two "lines"

        by_line = segment_characters_by_line(canvas)
        self.assertEqual(len(by_line), 2)
        self.assertEqual(len(by_line[0]), 2)
        self.assertEqual(len(by_line[1]), 2)

    def test_a_single_stray_pixel_is_filtered_as_noise(self):
        canvas = Image.new("L", (50, 50), color=255)
        canvas.putpixel((10, 10), 0)  # one lone dark pixel -- not a real character
        self.assertEqual(segment_characters(canvas), [])


if __name__ == "__main__":
    unittest.main()
