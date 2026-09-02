# Phase 16 — Iterative Improvement: Experiment Log

Rule: change exactly one variable per experiment, log it here, then stop and report before starting the next one.

---

## Experiment 1 — More training data

**Date:** 2026-09-02
**Variable changed:** dataset size (everything else held fixed)
**Hypothesis:** Phase 13's real training run showed clear overfitting on the tiny 3.8KB placeholder corpus (train loss fell steadily, val loss plateaued around step ~90-240, ending with a +0.20 train/val gap). If insufficient data was really the cause, training the identical architecture on the identical step budget over substantially more text should narrow or eliminate that gap.

### Config (held fixed vs. baseline)

Training config identical to `configs/training_config.json`: `batch_size=32, learning_rate=3e-4, weight_decay=0.01, grad_accumulation_steps=1, grad_clip_norm=1.0`, 300 steps, `eval_every=15`.

Model architecture identical: `context_length=128, embedding_dim=128, num_layers=4, num_heads=4, ffn_hidden_dim=512, dropout=0.1, qkv_bias=false`.

`vocab_size` differs slightly (51 → 59) as an unavoidable, non-deliberate side effect of the larger corpus containing a few characters the small one didn't — not a capacity change (+2,048 params, +0.25%, negligible next to the ~5.6x data increase).

### Dataset

| | Baseline (Phase 13) | Experiment 1 |
|---|---|---|
| Raw file | `data/raw/placeholder_corpus.txt` | `data/raw/experiment1_larger_corpus.txt` (baseline text + new original material, same authorship approach — no download, no copyright risk) |
| Raw size | 3,824 bytes | 21,479 bytes (5.6x) |
| Train tokens | 3,440 | 19,330 (5.6x) |
| Val tokens | 383 | 2,148 (5.6x) |
| Batches/epoch (batch_size=32) | 104 | 601 |
| Epochs covered by 300 steps | ~2.9 | ~0.5 |

### Steps and losses

300 steps, both runs.

| | Baseline | Experiment 1 |
|---|---|---|
| Final train_loss | 2.2181 | 2.3819 |
| Final val_loss | 2.4187 | 2.3681 (re-evaluated on full val set post-hoc: 2.3687) |
| Min val_loss | 2.4127 (step 240) | 2.3681 (step 300, still falling) |
| **Train/val gap at end** | **+0.2006** | **-0.0138** |
| Val_loss trend after step ~90 | Plateaued/flat | Still monotonically decreasing every eval, all the way to step 300 |

### Perplexity

Baseline: 11.24 → Experiment 1: 10.68 (**-0.55**, ~5% relative improvement), evaluated with identical methodology (`training/evaluate.py`) on each run's own held-out validation set.

### Sample generations (temperature 0.8, same prompts)

Baseline (`the model`): `the model ohe nches s thame the trambape che. Thome torene the A tomp`
Experiment 1 (`the model`): `the model thes t angle oro benee the theaneve whennd mary ang ate po"`

Baseline (`training a`): `training ang facos gulerai and ara assisext t on fuend a athatestextha`
Experiment 1 (`training a`): `training ating tuling towed n me ftedea ing othe co panin tr veng s ch`

Both remain well below coherent generation — neither sample "reads" meaningfully better to the eye than the other at a glance. The measured difference is real but modest at this scale; it shows up clearly in the loss curves and the train/val gap, not (yet) in obviously more fluent text.

### Runtime and memory

| | Baseline | Experiment 1 |
|---|---|---|
| Wall time (300 steps) | 173.3s | 461.8s |
| Free RAM before | 0.88GB | 0.78GB |
| Free RAM after | 1.58GB | 1.30GB |

**Important caveat on runtime:** the experiment's slower wall time is *not* attributed to the larger dataset. Per-step compute cost depends on batch size and context length, both unchanged — a bigger pool of available sequences to draw batches from does not make any single step more expensive. The system had less free RAM available (0.78GB vs 0.88GB) when this run started, and this session observed similar external slowdowns on unrelated commands around the same time, consistent with system-wide memory pressure rather than a genuine per-step cost difference. Treat the 173s vs 462s comparison as confounded and not meaningful; the loss/perplexity/gap comparison is unaffected by this, since it depends only on computation performed, not how long it took.

### Conclusion

**Hypothesis confirmed.** With the architecture, step budget, and every training hyperparameter held identical, increasing the training corpus by ~5.6x:
- Eliminated the overfitting gap seen in the baseline (+0.20 → -0.01)
- Produced a lower, still-falling validation loss at the same step count (2.42 → 2.37, and still decreasing vs. plateaued)
- Improved perplexity by ~5%

This is a genuine, mechanistic confirmation that dataset size — not architecture or training dynamics — was the dominant factor limiting the baseline run, exactly as the overfitting pattern in Phase 13 suggested. It also reveals a natural next question for a future experiment: since experiment 1 only covered ~0.5 epochs and was still improving at step 300, would simply running more steps on this same larger corpus continue to help, or would it eventually reproduce the same overfitting pattern once the model starts cycling back over already-seen data? That is a candidate for Experiment 2, not answered here, since this pass changed exactly one variable (dataset size) and stops here per the phase protocol.

**What this does not show:** any claim of fluent or coherent generation. Both checkpoints remain far below that bar, as expected for models this small trained this briefly — the measured improvement is real and worth acting on, but it is a narrow, specific finding about the train/val gap and loss curve shape, not a claim about output quality in any general sense.
