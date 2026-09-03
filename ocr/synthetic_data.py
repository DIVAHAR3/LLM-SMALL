"""Synthetic labeled character images for training an OCR classifier from
scratch -- no external dataset, no download. Each character in CHARACTERS
is rendered via Pillow using a curated set of TrueType fonts already
installed on this machine, at a few sizes, with small random
rotation/position jitter so the model doesn't just memorize one exact
pixel layout per character. The label is exact by construction (we chose
what to render), which is exactly why synthetic data is the standard
workaround when no labeled real-world dataset is available -- the same
"build our own small, local, legal corpus" approach this project already
took for its text data (Phase 3/4)."""
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ocr.normalize import IMAGE_SIZE, normalize_to_canvas

CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

FONT_DIR = Path(r"C:\Windows\Fonts")
FONT_NAMES = ["arial.ttf", "calibri.ttf", "times.ttf", "cour.ttf", "verdana.ttf", "tahoma.ttf"]
FONT_SIZES = [18, 20, 22]


def _available_fonts():
    """Only fonts that actually exist on THIS machine -- never assume a
    specific font is installed, same principle as never assuming
    hardware/CUDA is present."""
    return [FONT_DIR / name for name in FONT_NAMES if (FONT_DIR / name).exists()]


def render_character(char, font_path, font_size, rng, jitter_px=2, max_rotation_degrees=8):
    """One IMAGE_SIZE x IMAGE_SIZE grayscale image of `char`, tightly
    normalized via ocr.normalize.normalize_to_canvas -- the SAME
    function real-image segmentation (ocr/segment.py) crops its
    character regions through, so training data matches what the
    model actually sees at inference time. Rendered on an oversized
    canvas first (rotation can push ink outside a tightly-fitted one)
    with small random rotation and position jitter (deterministic
    given `rng`) before normalizing."""
    font = ImageFont.truetype(str(font_path), font_size)
    canvas_size = IMAGE_SIZE * 2  # generous headroom for rotation/jitter before the tight crop
    canvas = Image.new("L", (canvas_size, canvas_size), color=255)  # white background
    draw = ImageDraw.Draw(canvas)

    bbox = draw.textbbox((0, 0), char, font=font)
    char_w, char_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (canvas_size - char_w) // 2 - bbox[0] + rng.randint(-jitter_px, jitter_px)
    y = (canvas_size - char_h) // 2 - bbox[1] + rng.randint(-jitter_px, jitter_px)
    draw.text((x, y), char, font=font, fill=0)  # black ink

    angle = rng.uniform(-max_rotation_degrees, max_rotation_degrees)
    rotated = canvas.rotate(angle, resample=Image.BICUBIC, fillcolor=255)
    return normalize_to_canvas(rotated)


def generate_dataset(samples_per_character=30, seed=1337):
    """Returns a list of (PIL.Image, char) pairs, `samples_per_character`
    for every character in CHARACTERS. Deterministic given the same
    seed -- reproducible, same principle as Phase 24's seeded document
    splitting."""
    rng = random.Random(seed)
    fonts = _available_fonts()
    if not fonts:
        raise RuntimeError(
            f"no usable fonts found among {FONT_NAMES} in {FONT_DIR} -- cannot generate synthetic OCR data on this machine"
        )

    examples = []
    for char in CHARACTERS:
        for _ in range(samples_per_character):
            font_path = rng.choice(fonts)
            font_size = rng.choice(FONT_SIZES)
            image = render_character(char, font_path, font_size, rng)
            examples.append((image, char))
    return examples
