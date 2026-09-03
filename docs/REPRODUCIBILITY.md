# Phase 29 — Reproducibility

## What reproducibility actually requires

A seed alone doesn't make a run reproducible. Reproducing a result means being able to answer: what **code** produced it (git commit — the codebase changes over 29+ phases), what **data** did it train on (dataset version — Phase 24's pipeline can regenerate different splits from the same raw text depending on parameters), what **random draws** did it make (the seed), and what **software/hardware** did it run on (library versions can shift floating-point results even given an identical seed). All four have to be recorded together, or "reproducible" is an unverifiable claim.

**Where the seed has to be set matters.** Weight initialization (`nn.Linear`, `nn.Conv2d`, `nn.Embedding`, ...) draws from PyTorch's global RNG the moment a layer is *constructed* — not when training starts. Calling `torch.manual_seed()` inside a training loop, after the model already exists, can no longer make weight init reproducible; only later draws (DataLoader shuffling, dropout masks) would be affected. The seed has to be set in the calling script, before the model is built.

## A real gap found before writing any new code

Grepping the codebase for `manual_seed`/`random.seed` before starting this phase turned up something worth stating plainly: `configs/training_config.json` has had a `"seed": 1337` field since early in this project, but `torch.manual_seed()` was never actually called from any real training script or from `training/train.py`'s loop itself — only from tests (deterministic fixtures) and `inference/generate.py` (a `--seed` CLI flag for reproducible sampling). Every real training run so far — Phase 13's pretraining, Phase 26/27's fine-tuning, the OCR classifier — ran with an *unseeded* model initialization, config field notwithstanding. The field was recorded but never applied.

## What's new

**`training/reproducibility.py`** — `set_seed(seed)` seeds both Python's `random` (used directly by `training/data_prep.py`'s document splitting) and PyTorch's RNG. `capture_run_metadata(seed)` bundles the seed with:
- `get_git_commit()` — commit hash + whether the working tree was dirty at save time (`git rev-parse HEAD` / `git status --porcelain`, cached after the first call — see below).
- `get_environment_info()` — Python version, PyTorch version, platform string, CPU count, device (cpu/cuda).
- `get_dataset_info()` — reads `data/processed/meta.json` (Phase 24) if present, tying a checkpoint back to exactly which raw source file and preprocessing parameters produced its training data.

**`training/checkpoint.py`'s `save_checkpoint()`** now calls `capture_run_metadata()` automatically and stores the result under a new `"reproducibility"` key, extracting `seed` from the training config passed in (`config.get("seed")`). This happens for every checkpoint — GPT training, fine-tuning, and OCR training all go through this same function, so all three get this for free with no per-caller changes needed.

**Every real training script** (`scripts/train_ocr.py`, `scripts/finetune_instructions.py`, `scripts/finetune_chat_format.py`) now calls `set_seed()` explicitly, before constructing (or resizing, in the chat fine-tune script's case — `resize_vocab()` randomly initializes new embedding rows) its model. This closes the gap described above for every future run; it does not retroactively fix checkpoints already on disk.

## A real performance bug found by running the tests, not by design review

The first working version queried git via `subprocess.run()` on every single `save_checkpoint()` call. Individually each call is fast (~30-50ms measured directly), but Windows process-spawn overhead compounds badly when a test suite calls `save_checkpoint()` dozens of times — `tests/test_checkpoint.py` alone went from a fraction of a second to 74 seconds. Fixed by caching `get_git_commit()` with `functools.lru_cache`: the commit can't change mid-process anyway, so querying it once per process (not once per checkpoint save) is both correct and fast. Confirmed via a test asserting exactly 2 subprocess calls (not 6) across 3 calls to `get_git_commit()`.

## Verification — the actual stop condition, proven directly

Rather than only testing that metadata gets *recorded* correctly (a weaker claim), `tests/test_reproducibility.py`'s `TestEndToEndReproducibility` trains a fresh model **twice**, independently, calling `set_seed(2024)` before each run, and asserts the resulting weights are bit-for-bit identical — covering weight init, DataLoader shuffling order, and dropout masks together, not just one piece in isolation. A companion test with two *different* seeds confirms the models differ, ruling out a trivial "everything's always equal" false positive.

14 new tests in `tests/test_reproducibility.py`, 2 new in `tests/test_checkpoint.py` (the `reproducibility` key is present; its `seed` matches the training config). Full suite: 338/338 passing.
