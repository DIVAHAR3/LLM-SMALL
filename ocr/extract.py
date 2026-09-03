"""Combines segmentation (ocr/segment.py) and classification
(ocr/model.py's CharacterCNN, trained via scripts/train_ocr.py) into one
function: a full image in, predicted text out.

Segmentation only finds ink -- it has no idea where a space was, since a
space has no pixels to detect at all. Word breaks are reconstructed from
geometry instead: within a line, a horizontal gap between two consecutive
characters noticeably wider than that line's own MEDIAN gap is treated
as a word break -- an outlier-detection framing, not a fixed pixel
threshold, since gap sizes scale with font size. A new line (from
ocr.segment.segment_characters_by_line) becomes a newline.

Known limitation, verified directly rather than assumed: a line made up
ENTIRELY of single-character "words" (no multi-letter word anywhere in
it to establish what a normal within-word gap even looks like) has no
baseline to compare against, so this can fail to detect any of its
spaces at all. Real prose -- this pipeline's actual target -- almost
always has multi-letter words, so the common case works; an isolated
single-word test like "A B C" does not, an accepted gap in a
first-pass, from-scratch pipeline rather than something to overfit to.

Honest caveat, consistent with the rest of this project: this is a
small, from-scratch OCR pipeline with real, documented limitations --
see docs/OCR.md. It reads clean, isolated printed text reasonably; it
is not a production OCR engine."""
import torch

from ocr.dataset import INDEX_TO_CHAR, image_to_tensor
from ocr.segment import segment_characters_by_line

SPACE_GAP_MULTIPLIER = 2.0  # a gap more than this many times the line's
# own median inter-character gap is treated as a word break


def _predict_char(model, crop):
    tensor = image_to_tensor(crop).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
    return INDEX_TO_CHAR[logits.argmax(dim=1).item()]


def _line_to_text(model, line):
    if not line:
        return ""

    gaps = [line[i][1][0] - line[i - 1][1][2] - 1 for i in range(1, len(line))]
    space_threshold = 0
    if gaps:
        sorted_gaps = sorted(gaps)
        median_gap = sorted_gaps[len(sorted_gaps) // 2]
        space_threshold = SPACE_GAP_MULTIPLIER * median_gap

    chars = [_predict_char(model, line[0][0])]
    for i in range(1, len(line)):
        if gaps[i - 1] > space_threshold:
            chars.append(" ")
        chars.append(_predict_char(model, line[i][0]))
    return "".join(chars)


def extract_text(image, model):
    """image: a PIL.Image. model: a trained CharacterCNN in eval mode
    (see ocr.checkpoint.load_ocr_model_for_inference). Returns the
    predicted text, lines joined by "\\n"."""
    lines = segment_characters_by_line(image)
    return "\n".join(_line_to_text(model, line) for line in lines)
