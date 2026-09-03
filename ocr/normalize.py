"""Shared crop normalization used by BOTH the synthetic training-data
generator (ocr/synthetic_data.py) and real-image segmentation
(ocr/segment.py). Both MUST apply the exact same transformation to reach
CharacterCNN -- otherwise the model is trained on one visual scale/
padding convention and evaluated at inference time on a different one.

That's not a hypothetical concern: this project's own end-to-end testing
caught it directly. The first version of ocr/synthetic_data.py rendered
each character with generous natural whitespace inside its 28x28 canvas
(the ink filled roughly 43-54% of the frame), while ocr/segment.py's
first version tightly cropped a real image to its ink and resized that
crop to FILL the whole 28x28 frame -- a visibly different scale.
Real-sentence accuracy came out far below the reported validation
accuracy as a direct result. Routing both pipelines through this one
function is the fix: whatever scale/padding convention it encodes, both
training and inference now see the exact same thing."""
from PIL import Image, ImageOps

IMAGE_SIZE = 28
PADDING = 2


def normalize_to_canvas(image, padding=PADDING, target_size=IMAGE_SIZE, background=255):
    """Auto-detects `image`'s (mode "L", `background`-colored
    background, darker ink) ink bounding box, then normalizes it -- see
    normalize_bbox_to_canvas. Returns a blank canvas if the image has no
    ink at all. Use this when the caller does NOT already know the
    exact ink bounding box (e.g. ocr/synthetic_data.py, rendering one
    character onto an otherwise-blank canvas)."""
    pil_bbox = ImageOps.invert(image.convert("L")).getbbox()  # (left, upper, right, lower), right/lower EXCLUSIVE
    if pil_bbox is None:
        return Image.new("L", (target_size, target_size), color=background)
    left, upper, right, lower = pil_bbox
    inclusive_bbox = (left, upper, right - 1, lower - 1)  # -> (xmin, ymin, xmax, ymax), matching ocr/segment.py
    return normalize_bbox_to_canvas(image, inclusive_bbox, padding, target_size, background)


def normalize_bbox_to_canvas(image, bbox, padding=PADDING, target_size=IMAGE_SIZE, background=255):
    """Pads `bbox` (xmin, ymin, xmax, ymax; xmax/ymax INCLUSIVE, matching
    ocr/segment.py's connected-component boxes) by `padding` px, crops
    that region out of the FULL `image`, pads to a square (preserving
    aspect ratio so resizing doesn't distort the character), and
    resizes to target_size x target_size.

    Use this -- not normalize_to_canvas -- when the caller already knows
    the exact ink bounding box (ocr/segment.py, from connected-component
    analysis): cropping tightly to that box FIRST and only then calling
    normalize_to_canvas would leave no real surrounding pixels for the
    padding step to draw from, clamping padding to ~0 against the crop's
    own edges -- a real train/inference mismatch this project's own
    end-to-end testing caught. Padding here is taken from the original,
    uncropped image instead, the same way ocr/synthetic_data.py's
    renders always have real surrounding canvas to pad from."""
    xmin, ymin, xmax, ymax = bbox
    xmin = max(0, xmin - padding)
    ymin = max(0, ymin - padding)
    xmax = min(image.width, xmax + 1 + padding)
    ymax = min(image.height, ymax + 1 + padding)

    crop = image.crop((xmin, ymin, xmax, ymax))
    side = max(crop.width, crop.height)
    square = Image.new("L", (side, side), color=background)
    square.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    return square.resize((target_size, target_size), Image.LANCZOS)
