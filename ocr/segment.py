"""Classical text segmentation: locates individual character regions in
a full image via connected-component analysis, in reading order (top
line to bottom, left to right within each line), so each region can be
cropped and handed to CharacterCNN. No ML here -- binarization (Otsu's
method) and connected-component labeling (flood-fill/BFS) are both
deterministic, well-established classical algorithms, the same spirit
as analysis/image_analysis.py's median-cut color quantization.

Known limitation, stated plainly: this treats each connected group of
foreground pixels as one character. That's correct for most of
ocr/synthetic_data.py's CHARACTERS, but wrong for glyphs that are
naturally drawn as more than one disconnected stroke -- 'i' and 'j'
(dot + stem) will segment into two separate regions instead of one.
Not handled in this pass; a real limitation to fix later, not silently
worked around."""
from PIL import Image, ImageOps

from ocr.normalize import IMAGE_SIZE, normalize_bbox_to_canvas

MAX_DIMENSION = 800  # downsize before segmenting -- bounds the O(W*H) flood fill
MIN_COMPONENT_SIZE = 2  # px, in either dimension -- filters pure noise specks
MAX_COMPONENT_AREA_FRACTION = 0.5  # filters a mis-thresholded background blob
LINE_OVERLAP_FRACTION = 0.5  # min vertical overlap (of the shorter box) to join a line

_NEIGHBOR_OFFSETS = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]


def otsu_threshold(histogram):
    """Given a 256-bin grayscale histogram, returns the pixel value
    that best splits it into two groups by minimizing intra-class
    variance (equivalently, maximizing the variance BETWEEN the two
    groups) -- Otsu's method (1979): try every possible threshold,
    keep the one that best separates the histogram into two tight
    clusters rather than one that cuts through a populated region."""
    total = sum(histogram)
    if total == 0:
        return 0

    sum_total = sum(value * count for value, count in enumerate(histogram))
    weight_background, sum_background = 0, 0
    best_threshold, best_variance = 0, -1.0

    for value in range(256):
        weight_background += histogram[value]
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
        sum_background += value * histogram[value]

        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        between_class_variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

        if between_class_variance > best_variance:
            best_variance = between_class_variance
            best_threshold = value

    return best_threshold


def _ink_mask_and_normalized_image(grayscale):
    """Returns (mask, normalized_image): mask is a flat list of bool,
    True where a pixel is "ink" (foreground text), and normalized_image
    is the same picture with pixel VALUES flipped if needed so ink is
    always dark-on-light -- matching ocr/synthetic_data.py's rendering
    convention, which is what CharacterCNN was trained on. Polarity is
    decided by a simple, standard heuristic: whichever side of the
    Otsu threshold covers FEWER pixels is treated as the ink -- text is
    normally a minority of an image's pixels."""
    histogram = grayscale.histogram()
    threshold = otsu_threshold(histogram)
    dark_count = sum(histogram[: threshold + 1])
    light_count = sum(histogram[threshold + 1 :])
    pixels = list(grayscale.get_flattened_data())

    if dark_count <= light_count:
        mask = [p <= threshold for p in pixels]
        normalized = grayscale
    else:
        mask = [p > threshold for p in pixels]
        normalized = ImageOps.invert(grayscale)
    return mask, normalized


def find_connected_components(mask, width, height):
    """8-connected flood fill (iterative BFS, no recursion) over a flat
    boolean mask. Returns a list of (xmin, ymin, xmax, ymax) bounding
    boxes, one per connected group of foreground pixels."""
    visited = [False] * len(mask)
    boxes = []

    for start in range(len(mask)):
        if not mask[start] or visited[start]:
            continue
        visited[start] = True
        queue = [start]
        head = 0
        xmin = xmax = start % width
        ymin = ymax = start // width

        while head < len(queue):
            index = queue[head]
            head += 1
            x, y = index % width, index // width
            xmin, xmax = min(xmin, x), max(xmax, x)
            ymin, ymax = min(ymin, y), max(ymax, y)

            for dx, dy in _NEIGHBOR_OFFSETS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    neighbor_index = ny * width + nx
                    if mask[neighbor_index] and not visited[neighbor_index]:
                        visited[neighbor_index] = True
                        queue.append(neighbor_index)

        boxes.append((xmin, ymin, xmax, ymax))
    return boxes


def _filter_noise(boxes, image_width, image_height):
    image_area = image_width * image_height
    kept = []
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        w, h = xmax - xmin + 1, ymax - ymin + 1
        if w < MIN_COMPONENT_SIZE or h < MIN_COMPONENT_SIZE:
            continue
        if (w * h) / image_area > MAX_COMPONENT_AREA_FRACTION:
            continue
        kept.append(box)
    return kept


def group_into_reading_order(boxes):
    """Groups boxes into lines by vertical overlap, then returns them
    ordered top-to-bottom by line, left-to-right within each line."""
    lines = []  # each entry: list of boxes, roughly sharing a y-range
    for box in sorted(boxes, key=lambda b: b[1]):  # by ymin
        _, ymin, _, ymax = box
        box_height = ymax - ymin + 1
        placed = False
        for line in lines:
            line_ymin = min(b[1] for b in line)
            line_ymax = max(b[3] for b in line)
            line_height = line_ymax - line_ymin + 1
            overlap = min(ymax, line_ymax) - max(ymin, line_ymin)
            if overlap > LINE_OVERLAP_FRACTION * min(line_height, box_height):
                line.append(box)
                placed = True
                break
        if not placed:
            lines.append([box])

    lines.sort(key=lambda line: min(b[1] for b in line))
    ordered = []
    for line in lines:
        ordered.extend(sorted(line, key=lambda b: b[0]))
    return ordered


def _crop_and_prepare(normalized_image, box, target_size=IMAGE_SIZE):
    """Normalizes one already-known character bounding box against the
    FULL image via normalize_bbox_to_canvas -- the SAME underlying
    pad/square/resize logic ocr/synthetic_data.py's renders go through,
    so a real segmented crop ends up at the same visual scale/padding
    the model was trained on. Deliberately does NOT pre-crop to the
    tight box before normalizing (see normalize_bbox_to_canvas's
    docstring for why that would silently zero out the padding)."""
    return normalize_bbox_to_canvas(normalized_image, box, target_size=target_size)


def segment_characters(image):
    """End-to-end: a full image -> an ordered list of (crop, bbox)
    pairs, one per detected character region, in reading order. Each
    crop is IMAGE_SIZE x IMAGE_SIZE, grayscale, dark-ink-on-light --
    ready for CharacterCNN. bbox is in the (possibly downsized)
    working image's own coordinates."""
    grayscale = image.convert("L")
    if max(grayscale.size) > MAX_DIMENSION:
        scale = MAX_DIMENSION / max(grayscale.size)
        new_size = (max(1, round(grayscale.width * scale)), max(1, round(grayscale.height * scale)))
        grayscale = grayscale.resize(new_size, Image.LANCZOS)

    mask, normalized = _ink_mask_and_normalized_image(grayscale)
    boxes = find_connected_components(mask, grayscale.width, grayscale.height)
    boxes = _filter_noise(boxes, grayscale.width, grayscale.height)
    boxes = group_into_reading_order(boxes)

    return [(_crop_and_prepare(normalized, box), box) for box in boxes]
