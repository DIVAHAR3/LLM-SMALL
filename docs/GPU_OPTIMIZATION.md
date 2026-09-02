# Phase 23 — GPU Optimization

**This phase measures. The main codebase (`.venv`, training/inference) remains CPU-only, unchanged, as of this document.** The experiment ran in an isolated `.venv_gpu_test/` (gitignored) specifically so it couldn't disturb the working CPU environment regardless of outcome.

## The question this phase asked

Phase 2 chose CPU-only PyTorch, reasoning that the MX330's 2GB VRAM and old driver (certified for CUDA 11.4) made GPU support risky for uncertain benefit on models this tiny. Phase 23 revisited that decision empirically rather than carrying it forward unquestioned, now that real benchmark data (Phase 22) exists to compare against.

## Setup

Installed `torch==2.7.1+cu118` (2.62GB download, explicitly confirmed before downloading) into an isolated venv. CUDA **did** initialize — `torch.cuda.is_available()` returned `True`, device: `NVIDIA GeForce MX330`, compute capability `(6, 1)` (Pascal architecture). One real, practical cost worth noting: **first CUDA context initialization took over 2 minutes** — a one-time-per-process cost, but a genuinely slow one on this old driver/GPU combination, not something a couple of seconds' patience would have found.

## Finding 1: training-style batched compute — GPU wins substantially

Same methodology as Phase 22's compute-cost benchmark (fresh untrained models, forward+backward+optimizer.step(), proper `torch.cuda.synchronize()` around timing since CUDA ops are asynchronous — without it, timing would only measure how long it took to *queue* the work, not compute it).

| batch_size | context_length | CPU step_ms (Phase 22) | GPU step_ms | GPU speedup | peak VRAM |
|---:|---:|---:|---:|---:|---:|
| 8 | 64 | 53.8 | 50.4 | 1.07x | 52.5 MB |
| 32 | 64 | 195.8 | 48.6 | 4.03x | 124.3 MB |
| 8 | 128 | 128.2 | 51.2 | 2.50x | 80.5 MB |
| **32 | 128** (our actual Safe-tier training config) | **451.4** | **94.8** | **4.76x** | 239.0 MB |
| 32 | 256 | 1143.3 | 217.1 | 5.27x | 531.6 MB |
| 128 | 128 | *(not tested on CPU)* | 336.9 | — | 872.6 MB |
| 256 | 128 | *(not tested on CPU)* | **CUDA out of memory** | — | — |

At our actual training batch size (32), the GPU is **~4.76x faster** — a real, substantial, practical win, not a marginal one. My own prediction going into this (that kernel-launch overhead would likely dominate for a model this small) was **wrong** at batch_size ≥ 32, and only held at the smallest batch tested (8). This is exactly the value of measuring instead of reasoning from first principles alone.

**Real hard constraint confirmed**: `batch_size=256` at `context_length=128` OOM'd — 2GB VRAM is a genuine ceiling, not a theoretical one, consistent with what Phase 1 flagged as a risk.

## Finding 2: single-sequence autoregressive inference — GPU loses

Real trained checkpoint, real tokenizer, real generation loop — but batch=1 (one prompt at a time, which is what the actual API/frontend chat use case is), context growing token by token.

| max_new_tokens | CPU tokens/sec (Phase 22) | GPU tokens/sec | 
|---:|---:|---:|
| 20 | 264.7 | 58.4 |
| 100 | 183.0 | 62.1 |
| 300 | 126.0 | 69.7 |

Here the GPU is **1.8x to 4.5x *slower*** than CPU. This is the opposite workload shape from Finding 1: batch=1 with many small sequential steps means each step's actual GPU compute is tiny relative to fixed per-step costs (kernel launch, host↔device synchronization), so overhead dominates exactly as classical GPU-benchmarking intuition predicts — it just didn't apply to Finding 1's batched case. Interesting secondary detail: GPU tokens/sec slightly *increases* with longer generations (58→62→70), the opposite of CPU's trend (which falls due to the no-KV-cache growing-context cost documented in Phase 22) — plausibly because per-step overhead is fixed regardless of context size, so GPU's per-step cost stays roughly flat while some one-time costs amortize better over more steps. Not fully confirmed, noted honestly as the likely explanation rather than certain.

## Finding 3: mixed precision (FP16/AMP) — as predicted, not worth it

Compute capability `(6, 1)` means Pascal architecture, which lacks real Tensor Cores (a Volta/7.0+ feature) — consumer Pascal chips specifically have crippled FP16 throughput relative to FP32 by NVIDIA's own market segmentation. Measured rather than assumed:

```
FP32 (no AMP): 94.5 ms/step
FP16 (AMP):    90.3 ms/step
AMP speedup: 1.05x  (numerically stable, no NaN/divergence -- just not meaningfully faster)
```

A 5% difference is noise-level, not a real gain. **Recommendation: do not adopt mixed precision on this GPU** — it adds real complexity (gradient scaling, autocast contexts) for no measured benefit, precisely the "no precision change without stability check" caution the roadmap calls for. (The numerical stability check passed — loss stayed finite — but stability alone doesn't justify adopting something with no speed benefit.)

## What this justifies, and what it doesn't

- **Training**: the ~4.76x speedup at our actual batch size is real and substantial enough to be worth adopting for future training runs (re-running experiments, Phase 24+ work) — but that requires actually threading device-awareness through `training/train.py` and related code, which is a genuine implementation task, not something this measurement-only phase did. Deliberately left as a follow-up decision rather than silently implemented.
- **Inference (API/frontend)**: no change justified — CPU remains strictly better for the batch=1 chat use case Phase 18-21 built. Phase 2's original CPU-only choice stands for this workload.
- **Mixed precision**: not adopted anywhere. No measured benefit on this specific GPU.

This phase's own rule — "no blind optimization" — means these findings inform a future decision; they don't retroactively rewrite the codebase within this same phase.
