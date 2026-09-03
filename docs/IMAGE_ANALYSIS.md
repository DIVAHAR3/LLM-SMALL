# Image Analysis (Classical, No ML)

Not part of the numbered phase roadmap — a feature added by request between Phase 28 and Phase 29. Noted here and in `docs/ROADMAP.md` rather than silently folded into a phase number, so the roadmap stays an accurate record of what was actually built and when.

## Why this isn't "the model looking at a photo"

The GPT model this project trains is text-only and character-level — there is no path from a 128-dim token embedding to pixel data. Real image *understanding* (captioning, object recognition) would need a vision model trained on labeled images, which was explicitly ruled out for this feature: no external API, no other AI model, and no scope for a whole second from-scratch training effort right now.

What's genuinely buildable without any of that is **classical, deterministic analysis** — real, measured facts about an image's own pixels, computed once, with no training and no randomness. That's what this feature does. It does not claim to know what's *in* the photo.

## What gets measured

Given raw image bytes, `analysis/image_analysis.py`'s `analyze_image()` returns:

- **Dimensions & format** — width, height, aspect ratio, megapixels, format (PNG/JPEG/...), color mode, byte size.
- **Brightness & contrast** — mean and standard deviation of the grayscale-converted image. A flat, solid-color image has near-zero stddev; a high-contrast image has a large one.
- **Dominant colors** — via **median-cut color quantization**: a real, decades-old algorithm that repeatedly splits the image's color space along its widest axis until only `N` (5) representative colors remain, each weighted by the fraction of pixels closest to it. Downsampled to 150×150 first for speed; the resulting colors are the same either way, just cheaper to compute.
- **EXIF metadata** — camera make/model, timestamp, exposure settings, etc., *if the file actually embeds them* (most pasted screenshots and re-saved images won't). Binary/structural EXIF entries (GPS IFD pointers, MakerNote blobs) are deliberately excluded rather than guessed at or dumped as opaque bytes.
- **`ocr_text`** — text extracted by a separate, from-scratch OCR pipeline (segmentation + a trained CNN classifier, no external model), `null` if no OCR checkpoint is available on this server. See `docs/OCR.md` for the full story, including its real, honestly-documented limitations.

## Where it lives

- **`analysis/image_analysis.py`** — the pure analysis function, no FastAPI/HTTP dependency. 13 unit tests (`tests/test_image_analysis.py`) cover exact dimension/format reporting, brightness on solid vs. high-contrast images, dominant-color percentages on solid and half-and-half test images, EXIF extraction and its absence, grayscale/RGBA handling, and rejection of corrupt/empty bytes.
- **`POST /analyze/image`** (`api/main.py`) — multipart file upload, same auth (`X-API-Key`) and rate limiting as `/generate`. A separate, higher request-size limit (5 MB vs. the default 10 KB) is enforced via a `path_overrides` addition to `RequestSizeLimitMiddleware`, since a real photo is far larger than a short text prompt — every other route keeps the original 10 KB cap.
- **Frontend** — pasting an image into the chat input (`Ctrl+V` with an image on the clipboard) triggers a `paste` event handler in `App.jsx` that uploads it and renders the returned JSON in a dedicated "Image analysis" bubble, pretty-printed. Normal text chat is unaffected — the handler only intercepts the event when the clipboard actually contains an image; a plain text paste falls through to the browser's default behavior.

## New dependencies

- **Pillow** — image decoding and the classical operations (grayscale conversion, statistics, quantization, EXIF).
- **python-multipart** — FastAPI's required dependency for parsing `multipart/form-data` file uploads; without it, `UploadFile`/`File(...)` raise at request time.

Both are pinned in `requirements.txt`.

## Verification

- Backend: 13 tests in `tests/test_image_analysis.py` + 6 endpoint tests + 3 middleware-override tests in `tests/test_api.py`, all passing (full suite: 239/239).
- Frontend: 4 new tests in `App.test.jsx`/`api.test.js` (paste-to-analysis flow, error handling, ignoring non-image pastes, ignoring a paste while a request is already in flight), full suite 21/21 passing.
- Live end-to-end check: both dev servers started, driven with Playwright against real Chromium. A pasted 1×1 red-pixel PNG round-tripped correctly (`format: PNG`, `mode: RGBA`, `dominant_colors: [{hex: "#ff0000", percent: 100}]`, `exif: {}`), and a normal text prompt still streamed a reply correctly through the same shared code paths (`App.jsx`, `api.js`, the request-size-limit middleware). No console errors.

One real bug was caught during that live check, not by the unit tests: `handlePaste` had no `if (loading) return` guard at its top, unlike `handleSend`. In genuine browser use this was never reachable — a disabled `<input>` can't be focused, so a real `Ctrl+V` can't land on it while a previous request is in flight — but the test script exposed it by dispatching a synthetic `paste` event directly, bypassing that protection. Fixed to match `handleSend`'s existing guard, and covered by a new regression test asserting `analyzeImage` is never called while `loading` is true.
