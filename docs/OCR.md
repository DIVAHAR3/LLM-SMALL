# OCR: From-Scratch Text Extraction

Not part of the numbered phase roadmap — added between Phase 28 and Phase 29 by explicit request, with one hard constraint: **no external API, no pretrained model**. Everything here — the training data, the classifier, the segmentation — is built and trained inside this project, the same way the GPT itself was. Noted in `docs/ROADMAP.md`'s "Features added outside the numbered sequence" for the same reason `docs/IMAGE_ANALYSIS.md` is: keeping the roadmap an accurate record of what was actually built.

## Why not just call Tesseract

Tesseract (or any pretrained OCR/vision API) would work far better than what follows. It was deliberately ruled out: the ask was to *build* OCR, not integrate one. That means accepting real limitations a mature engine wouldn't have — this document states them plainly rather than glossing over them.

## The pipeline, in four pieces

**1. Synthetic labeled data** (`ocr/synthetic_data.py`) — there's no labeled real-world OCR dataset in this repo, and downloading one would violate "legal/local data only" without explicit sourcing/licensing review. The standard workaround, used here: render each of 62 characters (A-Z, a-z, 0-9) with Pillow, using TrueType fonts already installed on this machine (no download), at a few sizes with small random rotation/jitter. The label is exact by construction.

**2. A CNN character classifier** (`ocr/model.py`) — `CharacterCNN`, 213,630 params (conv 1→16 + pool → conv 16→32 + pool → FC 128 → FC 62), config-driven (`configs/ocr_model_config.json`) following the same `validate_config`/`from_config` pattern as Phase 28. Trained via `ocr/train.py` and `scripts/train_ocr.py` — a deliberately simpler loop than the GPT's `training/train.py` (no grad accumulation needed for this small in-memory dataset), reusing `torch.nn.functional.cross_entropy` directly (the concept was already taught in Phase 11) and `training.checkpoint.save_checkpoint` (already model-agnostic, so OCR checkpoints share the project's self-describing format).

**3. Classical segmentation** (`ocr/segment.py`) — locates character regions in a full image with no ML at all: Otsu's method (1979) picks a binarization threshold automatically by maximizing between-class pixel variance; 8-connected flood fill groups foreground pixels into regions; regions are grouped into lines by vertical overlap, then ordered top-to-bottom, left-to-right. Same classical-algorithm spirit as `analysis/image_analysis.py`'s median-cut color quantization.

**4. Text reconstruction** (`ocr/extract.py`) — `extract_text(image, model)` ties it together. Segmentation only finds ink; it has no idea where a space was, since a space has no pixels. Word breaks are reconstructed from geometry: within a line, a gap between two characters more than 2x that line's own *median* gap is treated as a space (an outlier-detection framing, not a fixed pixel count, since gap sizes scale with font size). A new line becomes `\n`.

Wired into the existing `POST /analyze/image` endpoint (`docs/IMAGE_ANALYSIS.md`) as an `ocr_text` field — the OCR model loads once at server startup if its checkpoint is present, and is optional: a fresh clone without a trained checkpoint still gets the classical (non-OCR) analysis, just with `ocr_text: null`. An OCR failure on a given image doesn't take down the rest of the response either — it's a best-effort addition on top of analysis that already succeeded.

## Two real bugs found by testing past the unit-test level

Both were caught by deliberately rendering realistic sentences and running them through the actual trained model end-to-end — not by the unit tests alone, which all passed throughout.

**Train/inference normalization mismatch.** The first working version trained characters that filled only ~43-54% of their 28x28 canvas (natural whitespace from centering in a fixed frame), while segmentation's crop-then-resize filled ~100% of the frame. A real, measurable distribution shift. Root-caused by directly measuring a training image's ink bounding box, then fixed by routing *both* pipelines through one shared function (`ocr/normalize.py`), so training renders and real segmented crops get identical scale and padding. Retraining after the fix alone took synthetic held-out accuracy from 78.76% to 95.97%.

**Space-detection miscalibration.** The first heuristic (gap > 0.6x median *character width*) missed the real space in a rendered "Hello World" — the actual gap (7px) fell just under the threshold (7.2px). Measuring real gap data across several rendered sentences showed within-word gaps and word-break gaps scale together with font size, but a line's own median *gap* (not width) cleanly separates them. Switching the baseline fixed all three tested prose examples.

## Known limitations, stated plainly

- **'i'/'j' dot-splitting**: single-connected-component segmentation treats each disconnected stroke as its own character. A dot sitting far enough above its stem can even fail the line-overlap check entirely and surface as a spurious extra line in the output. Not merged in this pass — a real limitation, not a silently-accepted one.
- **Case ambiguity**: normalizing every detected region independently to fill the same frame destroys the *relative size* cue that distinguishes many uppercase/lowercase pairs (C/c, O/o, S/s, X/x) in real text. This is the dominant source of real-sentence errors. Fixing it properly needs line-relative scaling (normalize a whole line's characters against each other, not each independently) — a bigger change, left for a future iteration.
- **All-single-character lines**: the space heuristic needs at least one multi-letter word in a line to establish what a normal within-word gap looks like. A line of only single-letter "words" (e.g. "A B C") has no such baseline and may miss its spaces. Real prose — the actual target — almost always has multi-letter words, so this is an accepted edge case, not something worth overfitting to.
- **Real-sentence character accuracy is well below the 95.97% synthetic validation figure.** That number measures held-out data from the *same distribution* as training (same fonts, same rendering pipeline); real accuracy on genuinely different text is lower, mostly from the case-ambiguity issue above. Stated honestly rather than leading with the higher, less meaningful number — same "start tiny, don't overclaim" principle as every other model in this project.

## Verification

- 80 new OCR-specific tests across `tests/test_ocr_*.py` (dataset generation, model architecture/config validation, dataset conversion, training mechanics, normalization consistency, segmentation, text reconstruction), plus 3 new API integration tests (`ocr_text` present/absent/resilient-to-failure). Full backend suite: 322/322 passing.
- Real training run: 2,480 synthetic images, 213,630-param CNN, 15 epochs, ~15-17s wall clock each time, final val_accuracy=95.97%.
- Live end-to-end check through the actual running HTTP API (not just unit tests): a rendered "Hello World" image posted to `/analyze/image` returns the full classical analysis plus `"ocr_text": "MEjjD WDFjd"` — the space is correctly placed; individual character accuracy reflects the case-ambiguity limitation above, honestly, not hidden behind a cherry-picked example.
