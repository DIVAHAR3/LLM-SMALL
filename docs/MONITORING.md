# Phase 30 — Monitoring

## Monitoring vs. evaluation

Phase 15's `training/evaluate.py` answers "how good is this checkpoint," computed *after* a run — val loss, perplexity, sample generations. Monitoring answers a different question: "what's happening *right now*, while training is running" — is loss actually decreasing, is the learning rate what I configured, how fast is it going, is memory under control. The distinction matters because monitoring is what lets you catch a problem (divergence, a stalled run, memory pressure) *while it's happening*, when you can still stop and fix it, rather than only after the run finishes.

## What gets logged

At the same cadence validation already runs at (`eval_every` — there was no reason to put these on separate schedules), `training/train.py`'s `train()` now logs:

- **train/val loss** (already existed)
- **learning rate** — the optimizer's actual current value (`optimizer.param_groups[0]["lr"]`), not just what was configured. No LR schedule exists yet (that's Phase 35 territory), so this is currently constant within a run — but recording the *actual* value now means it stays correct automatically once a schedule is added later.
- **tokens processed** — a cumulative counter, incremented by each micro-batch's real token count (`y.numel()`), independent of gradient-accumulation window structure.
- **elapsed time** — wall-clock seconds since `train()` was called.
- **available memory** — reusing the Windows `ctypes`-based query, relevant given this project's own hardware (often under 1GB free — see `docs/phase1_inspection_report.md`).

Console output, one line per eval point:
```
step 20: train_loss=2.9698 val_loss=2.9984 lr=3.00e-03 tokens=2,560 elapsed=0.3s mem_avail=782MB
```

## Optional lightweight file logging — a deliberate design choice, not full TensorBoard

The roadmap calls for "optional lightweight TensorBoard-style logging." Read literally, that could mean pulling in the actual `tensorboard` package — but it drags in a substantial dependency chain (protobuf, grpcio, werkzeug, absl-py, and more) for a project that has deliberately kept its dependency list minimal throughout (no numpy, even). The chosen alternative: pass `log_path` to `train()` and each eval-point's metrics get appended as one JSON line (JSONL) to that file — structured, one record per logged step, trivially loadable into pandas/matplotlib later for exactly the same kind of over-time plots TensorBoard would show, without the dependency. `log_path` is optional and defaults to `None` (no file written) — every existing call site is unaffected.

**Option A (chosen):** JSONL file, stdlib only, zero new dependencies.
**Option B (not chosen):** Real `tensorboard.SummaryWriter` integration — richer UI (live web dashboard, image/histogram logging), but a real dependency addition this project doesn't otherwise need. Worth revisiting if training runs grow long/complex enough that a live dashboard earns its cost — not needed at this project's current scale.

Appends rather than overwrites, so a resumed run's history isn't lost by resuming into the same file; delete/rotate the log path yourself before a genuinely fresh run if mixing old and new runs in one file isn't wanted.

## A refactor along the way

`get_available_memory_mb()` existed only in `scripts/benchmark.py` (Phase 22). Moved to a new `training/monitoring.py` so `training/train.py` could reuse it without `training/` (library code) depending on `scripts/` (entry-point scripts) — the wrong direction. `scripts/benchmark.py` now imports it from there instead; behavior and its existing test are unchanged.

## Verification

A real short training run (20 steps, `eval_every=5`, tiny model) with `log_path` set, reviewed directly — not just asserted to look right:

```
step  5: train_loss=3.0629 val_loss=3.0569 lr=3.00e-03 tokens=640    elapsed=0.1s mem_avail=773MB
step 10: train_loss=3.1207 val_loss=3.0272 lr=3.00e-03 tokens=1,280  elapsed=0.2s mem_avail=774MB
step 15: train_loss=3.0937 val_loss=3.0172 lr=3.00e-03 tokens=1,920  elapsed=0.2s mem_avail=779MB
step 20: train_loss=2.9698 val_loss=2.9984 lr=3.00e-03 tokens=2,560  elapsed=0.3s mem_avail=782MB
```

`tokens_processed` matches the expected count exactly (8 batch_size × 16 context_length × 5 steps = 640 tokens per interval), elapsed time and memory both move sensibly, and the JSONL file's content matches the console output line for line, confirming file and console logging stay consistent with each other.

10 new tests (`tests/test_train.py`'s `TestMonitoring`: 9; `tests/test_monitoring.py`: 1 moved-and-preserved test — plus existing coverage unaffected). Full suite: 348/348 passing.
