# Phase 28 — Config-Driven Architecture

## The problem with scattered constants

Architecture dimensions (`vocab_size`, `context_length`, `embedding_dim`, `num_layers`, `num_heads`, `ffn_hidden_dim`, `dropout`) determine the exact shape of every weight matrix in the model. If the same numbers are typed as literals at more than one call site, they can drift out of sync — change the real architecture in `configs/model_config.json` and forget a hardcoded copy elsewhere, and you get code that silently builds the *wrong* model, or a benchmark that compares against an architecture nobody actually trained. A single source of truth removes that failure mode entirely: every real code path reads the same file, so there is exactly one place to change.

The second half of this phase — validating a config *before* constructing anything — exists for a different reason: catching a bad config (e.g. `embedding_dim` not divisible by `num_heads`) as one clear error message up front is much better than the confusing shape mismatch that would otherwise surface deep inside `MultiHeadAttention`, after most of the model has already been built.

## Where this project already stood

Most of the codebase was already config-driven going into this phase — `GPTModel.from_config()` (added in Phase 10) already existed, and `training/evaluate.py` and `training/checkpoint.py` already used it. `api/main.py` never constructs a model directly at all — it loads a checkpoint, and self-describing checkpoints (Phase 17) carry their own `model_config`, so the server is architecture-agnostic by construction.

A project-wide grep for hardcoded architecture literals (`embedding_dim=128`, `num_layers=4`, etc.) outside test files turned up exactly one real gap: `scripts/benchmark.py`'s `benchmark_compute_cost()` took `vocab_size`, `embedding_dim`, `num_layers`, `num_heads`, and `ffn_hidden_dim` as hardcoded function defaults, duplicating `configs/model_config.json`'s values as separate literals.

(Test files construct `GPTModel(...)` directly with small literal dimensions on purpose — fast, self-contained unit tests shouldn't depend on an external config file. That's normal test practice, not the kind of drift risk this phase targets.)

## What changed

**`model/gpt.py`** — added `validate_model_config(config)`: checks that all required architecture keys are present, are positive integers (not bools, not floats), that `dropout` is a number in `[0, 1)`, and that `embedding_dim` is divisible by `num_heads`. `GPTModel.from_config()` now calls it before building anything, since `from_config()` is the intended entry point for every real model construction in the project.

**`scripts/benchmark.py`** — `benchmark_compute_cost()` now takes a `model_config` dict instead of five separate literal-defaulted parameters. The benchmark still sweeps `batch_size` and `context_length` explicitly (that's the actual experiment), but every other dimension comes from the same `configs/model_config.json` every other real code path uses, loaded once in `main()` via a new `--model-config` flag.

## Verification

Full suite: 217/217 passing (208 before this phase + 9 new: 8 for `validate_model_config` covering missing keys, non-positive values, non-integer values, out-of-range dropout, the `embedding_dim`/`num_heads` divisibility check, the real `configs/model_config.json` passing validation, and `from_config` rejecting a bad config before construction; 1 for `benchmark_compute_cost`'s new `context_length`-argument-overrides-config behavior).

No hardcoded architecture constants remain in non-test code — `configs/model_config.json` is the only place architecture dimensions are defined for real runs, and an invalid config now fails fast with a specific, readable error instead of a confusing failure inside a submodule.
