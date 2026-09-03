# Roadmap — GPT-from-Scratch

Work exactly one phase at a time per the protocol in `CLAUDE.md`. Each entry below is the target for that phase; teach the underlying concept before writing the code for it.

| # | Phase | Deliverable | Stop condition |
|---|-------|-------------|-----------------|
| 1 | Server inspection | Run read-only inspection commands (OS, CPU, RAM, disk, GPU/nvidia-smi, CUDA, PyTorch presence, git, docker, network reachability to pypi). No installs, no downloads. | Report + 3 model-size proposals (Safe / Balanced / Maximum-experimental) with params, vocab size, context length, embed dim, layers, heads, batch size, LR, precision, est. memory. Recommend one. |
| 2 | Environment setup | venv, `requirements.txt`, `.gitignore`, base `README.md`, config files, git init. Show package list before installing. | Verified `python`, `pip`, `torch` import (+ CUDA check if applicable). |
| 3 | Tokenizer | Explain tokens/vocab/encoding/special & unknown tokens/context length. Implement a simple (e.g. char or whitespace) tokenizer: `encode`, `decode`, `vocab_size`, `save`, `load`. | Round-trip test on "Hello world" passes. |
| 4 | Dataset | Explain next-token prediction. Build raw→clean→tokenize→split→sequence→`Dataset`/`DataLoader` pipeline. Configurable `context_length`, `batch_size`, split ratio. Small legal/local dataset only. | Dataset tests pass on a tiny corpus. |
| 5 | Token + positional embeddings | Explain why token IDs alone lack meaning. Implement embedding lookup + positional embedding, show token_emb + pos_emb = model input. | Tensor shape tests pass. |
| 6 | Self-attention | Teach Q/K/V, scaled dot-product attention, softmax, causal masking. Implement single-head, multi-head, causal mask. | Tests confirm shapes, masking blocks future tokens, multi-head works. |
| 7 | Feed-forward network | Explain FFN role in a transformer block. Implement Linear→Activation→Linear, configurable dims. | Shape tests pass. |
| 8 | LayerNorm + residuals | Explain normalization and residual/skip connections. Implement both independently. | Unit tests pass. |
| 9 | Transformer block | Combine LN → MHSA → residual → LN → FFN → residual into one configurable block. | Forward pass, shape, causality, and gradient tests pass. |
| 10 | Full GPT model | Stack blocks: embeddings → N transformer blocks → final LN → LM head → logits `(B, T, vocab_size)`. Configurable vocab_size/context_length/embedding_dim/num_layers/num_heads/dropout. | Param count reported, model summary produced, random-input test passes. |
| 11 | Loss function | Explain cross-entropy for next-token prediction. Implement it. | Loss visibly decreases on a tiny synthetic dataset. |
| 12 | Training loop | AdamW optimizer, LR, epochs, batch size, grad accumulation, grad clipping, train/val loss logging, checkpointing, resume. Checkpoint = model state + optimizer state + epoch/step + config + metrics. | Smoke test only — no long run. |
| 13 | First real training run | Report params, dataset size/tokens, batch size, context length, expected steps/runtime/memory *before* starting. Run a short training run only, monitor loss/val loss/mem/throughput. | Short run completes; explain results. |
| 14 | Text generation | Autoregressive loop: prompt→tokenize→model→logits→sample→append→repeat. Support greedy, temperature, top-k, top-p, max_new_tokens, stop tokens. CLI: `python -m inference.generate --prompt "Hello"`. | Generation test passes; explain why a tiny model generates weak text. |
| 15 | Evaluation | Track train/val loss, perplexity, sample generations, param count, tokens processed, speed, memory. Simple report format. | Metrics explained; checkpoints compared if >1 exists. |
| 16 | Iterative improvement | Change **one variable at a time** from: tokenizer, dataset size/cleaning, more tokens, larger context/embed dim/layers/heads, LR schedule, warmup, weight decay, grad accumulation, mixed precision, sampling. Log each as an experiment (config, dataset, steps, losses, perplexity, generations, runtime, memory, conclusion). | One experiment logged per phase pass. |
| 17 | Checkpoint management | Save/load/resume/inference-load. Checkpoint must be self-describing (config included). Verify reload reproduces identical behavior. Never commit checkpoints. | Reload-equivalence test passes. |
| 18 | FastAPI inference server | `GET /health`, `POST /generate` with `{prompt, max_new_tokens, temperature, top_k, top_p}` → `{text}`. Validation, error handling, logging, configurable model path/device. Local only (`127.0.0.1`). | `curl` health check + one generate call succeed locally. |
| 19 | API security | Explain auth, API keys, rate limiting, request-size limits, input validation, CORS, HTTPS, reverse proxy, logging, abuse prevention. Secrets via `.env`/`.env.example` only. | Written security plan reviewed before any public exposure. |
| 20 | React frontend | Vite app: chat input, send button, history, user/assistant bubbles, loading + error states, clear button, `VITE_API_URL` env var. | Frontend talks to local FastAPI end-to-end. |
| 21 | Streaming | Add SSE (or similar) token streaming from API to browser, only after non-streaming generation is stable. | Incremental rendering verified in browser. |
| 22 | Benchmarking | Script measuring tokens/sec, latency, first-token latency, CPU/GPU/RAM usage, load time; compare CPU vs GPU, batch sizes, context lengths. | Benchmark results reported, no blind optimization. |
| 23 | GPU optimization | If GPU present: mixed precision (FP16/BF16), grad scaling, grad accumulation, DataLoader tuning, pinned memory, batch size, CUDA sync for accurate timing. | Before/after numbers reported; no precision change without stability check. |
| 24 | Data pipeline hardening | Dedup, malformed-text handling, normalization, reproducible split, token counting, dataset stats report (docs/chars/tokens/vocab/train-val tokens/avg doc length). | Stats report produced. Only legal/authorized data. |
| 25 | Subword tokenizer | Explain BPE, vocab size tradeoffs, token efficiency, unknown-token handling. Compare against Phase 3 tokenizer on real text. | Side-by-side token-count comparison; pick one with justification. |
| 26 | Instruction tuning | Explain pretraining vs. instruction tuning. Build a small legal instruction dataset, convert to training format, fine-tune base model, evaluate before/after. | Before/after generations compared. |
| 27 | Chat format | Implement `<system>/<user>/<assistant>` structured format, train/adapt on it, add chat-mode inference. | Chat-formatted generation works. |
| 28 | Config-driven architecture | All architecture constants live in `configs/model_config.json` (vocab_size, context_length, embedding_dim, num_layers, num_heads, dropout). Validate config before model creation. | No hardcoded architecture constants remain in code. |
| 29 | Reproducibility | Seed control; record git commit, model/training config, dataset version, seed, hardware, PyTorch/CUDA versions per run. | A run can be reproduced from its recorded metadata. |
| 30 | Monitoring | Log step, train/val loss, LR, tokens processed, elapsed time, memory. Optional lightweight TensorBoard-style logging. | Monitoring output reviewed for one run. |
| 31 | Deployment plan | Document Internet → HTTPS → reverse proxy → FastAPI → LLM path. Explain process management, restarts, logs, health checks, resource limits. No public exposure without approval. | Written deployment plan approved. |
| 32 | Docker | `Dockerfile` + `docker-compose.yml` for API (+frontend if useful). Verify GPU passthrough if needed. Don't containerize training unless clearly beneficial. | Container runs and serves `/health`. |
| 33 | Model versioning | `models/<name>-v<n>/` directories; record architecture, param count, tokenizer/dataset version, config, val loss, perplexity, date, git commit per version. | Versioning scheme in place with ≥1 entry. |
| 34 | Scaling analysis | Re-inspect hardware. Estimate memory (model/optimizer/activations) and data/training difficulty for 10M/50M/100M/500M/1B param models. No auto-scaling. | Recommendation given based on actual hardware. |
| 35 | Advanced concepts (teach progressively) | KV cache, RoPE, RMSNorm, SwiGLU, weight tying, LR schedules, AdamW internals, mixed precision, gradient checkpointing, data/tensor/pipeline parallelism, quantization, LoRA/QLoRA, preference optimization. Explain each: what/why/math/vs-current-impl; implement only when it clearly helps. | Concept explained; implement selectively. |
| 36 | Production inference | KV cache, batching, quantization, faster load, memory management, request queueing, concurrency. Benchmark every change. | Before/after benchmarks for each optimization. |
| 37 | Final documentation | Full README: what/how-LLMs-work/architecture/tokenizer/dataset/training/generation/eval/API/frontend/deployment/hardware/config/troubleshooting/security/future work. | README covers all sections. |
| 38 | Final system test | Run full pipeline end-to-end (data→tokenizer→dataset→model→train→checkpoint→reload→generate→API→frontend). Run all tests. Confirm no secrets exposed, no unexpected file changes. | Final system report produced. |

## Notes carried over from the original spec
- Never treat a small model as production-equivalent.
- Legal/local data only — no copyrighted or private data without authorization.
- Every phase ends with a stop; resume only on `continue`.

## Features added outside the numbered sequence

Occasionally a feature request doesn't fit any single numbered phase above. Rather than silently renumbering the roadmap or stretching an existing phase's scope, these are tracked here explicitly, in the order they were added, each following the same protocol (concept taught, tests, report, commit) as a numbered phase.

- **Image analysis (classical, no ML)** — added between Phase 28 and Phase 29. Paste an image into the chat UI, get back deterministic, measured properties (dimensions, format, brightness/contrast, dominant colors via median-cut quantization, EXIF) as JSON — no model, no training, no external API, per explicit request. See `docs/IMAGE_ANALYSIS.md`.
- **OCR (from-scratch text extraction)** — added immediately after, same "no API, no other AI" constraint. A small CNN character classifier trained on synthetic (self-rendered, locally-generated) data, combined with classical connected-component segmentation, wired into the image-analysis JSON as `ocr_text`. Real, honestly-documented limitations (case ambiguity, i/j dot-splitting) rather than an inflated accuracy claim. See `docs/OCR.md`.
