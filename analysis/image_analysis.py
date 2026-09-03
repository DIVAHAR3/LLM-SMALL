"""Classical, deterministic image analysis: no model, no training, no
external calls. Every field returned here is a directly measured property
of the image's own pixels (dimensions, brightness, color composition,
embedded metadata) -- this deliberately does NOT attempt to say what the
image depicts. That would require a trained vision model; this project's
"no API, no other AI" constraint rules that out for now (see
docs/IMAGE_ANALYSIS.md)."""
from io import BytesIO

from PIL import ExifTags, Image, ImageStat, UnidentifiedImageError
from PIL.TiffImagePlugin import IFDRational

DOMINANT_COLOR_COUNT = 5
_QUANTIZE_SAMPLE_SIZE = (150, 150)  # downsample before quantizing -- same
# handful of dominant colors come out, far fewer pixels to process

# Structural EXIF entries (IFD pointers, opaque binary blobs) rather than
# actual photo data -- including them as plain values would be misleading.
_SKIP_EXIF_TAGS = {"GPSInfo", "ExifOffset", "MakerNote", "UserComment"}


def analyze_image(image_bytes):
    """Returns a JSON-serializable dict of measured properties for the
    given raw image bytes. Raises ValueError if the bytes aren't a
    decodable image."""
    try:
        image = Image.open(BytesIO(image_bytes))
        image.load()  # force full decode now, not lazily on first use --
        # a truncated/corrupt file should fail here, not mid-computation
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"could not decode image: {exc}") from exc

    width, height = image.size
    return {
        "format": image.format,
        "mode": image.mode,
        "width": width,
        "height": height,
        "aspect_ratio": round(width / height, 4),
        "size_bytes": len(image_bytes),
        "megapixels": round(width * height / 1_000_000, 2),
        "brightness": _brightness_stats(image),
        "dominant_colors": _dominant_colors(image),
        "exif": _safe_exif(image),
    }


def _brightness_stats(image):
    grayscale = image.convert("L")
    stat = ImageStat.Stat(grayscale)
    return {"mean": round(stat.mean[0], 2), "stddev": round(stat.stddev[0], 2)}


def _dominant_colors(image, count=DOMINANT_COLOR_COUNT):
    """Median-cut color quantization: repeatedly splits the image's color
    space along its widest axis until only `count` representative colors
    remain, weighted by how many pixels fall closest to each one."""
    rgb = image.convert("RGB")
    rgb.thumbnail(_QUANTIZE_SAMPLE_SIZE)
    quantized = rgb.quantize(colors=count, method=Image.Quantize.MEDIANCUT)
    counts = quantized.convert("RGB").getcolors(maxcolors=rgb.width * rgb.height)
    counts.sort(key=lambda item: item[0], reverse=True)
    total_pixels = sum(pixel_count for pixel_count, _ in counts)
    return [
        {"hex": "#{:02x}{:02x}{:02x}".format(*color), "percent": round(100 * pixel_count / total_pixels, 1)}
        for pixel_count, color in counts
    ]


def _safe_exif(image):
    """Only well-known, JSON-safe scalar EXIF tags -- skips anything
    binary or structural (MakerNote, GPS IFD pointers, raw thumbnails)
    that isn't safe to hand straight to json.dumps()."""
    exif = image.getexif()
    if not exif:
        return {}

    result = {}
    for tag_id, value in exif.items():
        name = ExifTags.TAGS.get(tag_id, str(tag_id))
        if name in _SKIP_EXIF_TAGS:
            continue

        if isinstance(value, IFDRational):
            value = float(value)
        elif isinstance(value, bytes):
            try:
                value = value.decode("ascii", errors="ignore").strip("\x00").strip()
            except Exception:
                continue
        elif isinstance(value, str):
            value = value.strip("\x00").strip()

        if isinstance(value, (str, int, float)) and value != "":
            result[name] = value
    return result
