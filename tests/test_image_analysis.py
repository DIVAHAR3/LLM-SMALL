import sys
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.image_analysis import analyze_image


def make_image_bytes(image, fmt="PNG", **save_kwargs):
    buffer = BytesIO()
    image.save(buffer, format=fmt, **save_kwargs)
    return buffer.getvalue()


class TestAnalyzeImage(unittest.TestCase):
    def test_reports_correct_dimensions_format_and_mode(self):
        image = Image.new("RGB", (40, 20), color=(255, 0, 0))
        result = analyze_image(make_image_bytes(image, "PNG"))
        self.assertEqual(result["width"], 40)
        self.assertEqual(result["height"], 20)
        self.assertEqual(result["format"], "PNG")
        self.assertEqual(result["mode"], "RGB")

    def test_aspect_ratio_and_megapixels_are_computed_correctly(self):
        image = Image.new("RGB", (200, 100), color=(0, 0, 0))
        result = analyze_image(make_image_bytes(image))
        self.assertAlmostEqual(result["aspect_ratio"], 2.0)
        self.assertAlmostEqual(result["megapixels"], 0.02)

    def test_size_bytes_matches_actual_encoded_length(self):
        image = Image.new("RGB", (10, 10), color=(1, 2, 3))
        raw = make_image_bytes(image)
        result = analyze_image(raw)
        self.assertEqual(result["size_bytes"], len(raw))

    def test_solid_color_image_has_near_zero_brightness_stddev(self):
        image = Image.new("RGB", (30, 30), color=(128, 128, 128))
        result = analyze_image(make_image_bytes(image))
        self.assertAlmostEqual(result["brightness"]["mean"], 128, delta=1)
        self.assertAlmostEqual(result["brightness"]["stddev"], 0, delta=0.5)

    def test_solid_color_image_has_one_dominant_color_at_100_percent(self):
        image = Image.new("RGB", (30, 30), color=(10, 200, 50))
        result = analyze_image(make_image_bytes(image))
        colors = result["dominant_colors"]
        self.assertEqual(len(colors), 1)
        self.assertEqual(colors[0]["hex"], "#0ac832")
        self.assertEqual(colors[0]["percent"], 100.0)

    def test_half_black_half_white_image_splits_roughly_evenly(self):
        image = Image.new("RGB", (40, 40), color=(0, 0, 0))
        for x in range(20, 40):
            for y in range(40):
                image.putpixel((x, y), (255, 255, 255))
        result = analyze_image(make_image_bytes(image))
        colors = result["dominant_colors"]
        self.assertEqual(len(colors), 2)
        percents = sorted(c["percent"] for c in colors)
        self.assertAlmostEqual(percents[0], 50.0, delta=2)
        self.assertAlmostEqual(percents[1], 50.0, delta=2)
        # high-contrast image -> high stddev, unlike the solid-color case
        self.assertGreater(result["brightness"]["stddev"], 100)

    def test_dominant_colors_percentages_sum_to_100(self):
        image = Image.new("RGB", (25, 25), color=(0, 0, 0))
        for x in range(25):
            for y in range(25):
                image.putpixel((x, y), (x * 10 % 256, y * 10 % 256, 0))
        result = analyze_image(make_image_bytes(image))
        self.assertAlmostEqual(sum(c["percent"] for c in result["dominant_colors"]), 100.0, delta=0.5)
        self.assertLessEqual(len(result["dominant_colors"]), 5)

    def test_image_without_exif_returns_empty_exif_dict(self):
        image = Image.new("RGB", (10, 10), color=(5, 5, 5))
        result = analyze_image(make_image_bytes(image, "PNG"))
        self.assertEqual(result["exif"], {})

    def test_exif_tags_are_extracted_when_present(self):
        image = Image.new("RGB", (10, 10), color=(5, 5, 5))
        exif = Image.Exif()
        exif[271] = "TestCamera"  # 271 = Make
        exif[272] = "Model X"  # 272 = Model
        raw = make_image_bytes(image, "JPEG", exif=exif)

        result = analyze_image(raw)
        self.assertEqual(result["exif"].get("Make"), "TestCamera")
        self.assertEqual(result["exif"].get("Model"), "Model X")

    def test_grayscale_image_is_handled(self):
        image = Image.new("L", (16, 16), color=200)
        result = analyze_image(make_image_bytes(image))
        self.assertEqual(result["mode"], "L")
        self.assertEqual(len(result["dominant_colors"]), 1)

    def test_rgba_image_is_handled(self):
        image = Image.new("RGBA", (16, 16), color=(10, 20, 30, 128))
        result = analyze_image(make_image_bytes(image))
        self.assertEqual(result["mode"], "RGBA")
        self.assertEqual(len(result["dominant_colors"]), 1)

    def test_invalid_bytes_raise_value_error(self):
        with self.assertRaises(ValueError):
            analyze_image(b"this is not an image")

    def test_empty_bytes_raise_value_error(self):
        with self.assertRaises(ValueError):
            analyze_image(b"")


if __name__ == "__main__":
    unittest.main()
