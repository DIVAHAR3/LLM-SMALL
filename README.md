# GPTbot


An educational, decoder-only GPT-style Transformer built in PyTorch from scratch, trained on this machine, and eventually served via FastAPI with a React/Vite chat UI.

This is a learning project, not a production system — see `CLAUDE.md` for the governing rules and `docs/ROADMAP.md` for the full phase-by-phase plan.

## Status

Completed through **Phase 31 — Deployment plan** (`docs/DEPLOYMENT_PLAN.md`, reviewed and approved: the full Internet → HTTPS → reverse proxy → FastAPI → LLM path, process management, restarts, health checks, and resource limits — writing the plan does not itself authorize deployment; the server stays bound to `127.0.0.1` regardless. Found and fixed a real gap while writing it: `logs/api.log` had no rotation and would have grown unbounded on a real deployed server — already ~580KB from local dev/test use alone — switched to `logging.handlers.RotatingFileHandler`). Phase 30 — monitoring — added training logs for learning rate, cumulative tokens processed, elapsed time, and available memory alongside train/val loss, at the same cadence validation already runs at; optional lightweight JSONL file logging — chosen deliberately over a full TensorBoard dependency this project doesn't otherwise need — see `docs/MONITORING.md`). Phase 29 made every checkpoint self-describe how it could be reproduced — git commit, dataset provenance, environment, and seed — and found a real gap along the way: `training_config.json`'s `seed` field was recorded but never actually applied by any real training run until then (`docs/REPRODUCIBILITY.md`). Phase 28 added `validate_model_config()` so a bad architecture config fails clearly before any submodule is built (`docs/CONFIG_DRIVEN_ARCHITECTURE.md`); Phase 27 added `<SYSTEM>`/`<USER>`/`<ASSISTANT>` chat format as atomic special tokens with safe vocabulary/embedding extension (`docs/CHAT_FORMAT.md`); Phase 26 — instruction tuning — fine-tuned the pretrained model on a small original instruction dataset (a measurable statistical shift toward the new data confirmed; genuine instruction-following not expected or claimed at this scale — see `docs/INSTRUCTION_TUNING.md`). Phase 25 added a from-scratch BPE subword tokenizer (~45% fewer tokens than char-level on real project text at vocab_size=300 — `docs/TOKENIZER_COMPARISON.md`; recommended for future training, not yet cut over since it requires retraining), and Phase 24 hardened the data pipeline (dedup, Unicode normalization, reproducible document-level splitting — `docs/DATASET_STATS.md`). Full decoder-only GPT (embeddings → 4 transformer blocks → final LayerNorm → LM head, 821,248 params, Safe tier) trains end-to-end on CPU, checkpoints/resumes correctly, generates text via `python -m inference.generate --prompt "..."` (greedy, temperature, top-k, top-p, stop tokens all supported), can be evaluated/compared via `training/evaluate.py` (val loss, perplexity, sample generations, throughput), is served locally over HTTP via FastAPI (`GET /health`, `POST /generate`, `POST /generate/stream` for token-by-token SSE streaming), and has a local Vite + React chat UI (`frontend/`) that renders responses incrementally as they stream in. See `docs/phase1_inspection_report.md` for the original hardware findings and model-size tiers, `docs/BENCHMARKS.md` for measured CPU baselines, and `docs/GPU_OPTIMIZATION.md` for a measured CPU-vs-GPU comparison (training ~4.76x faster on GPU; inference actually slower on GPU — main codebase stays CPU-only pending a follow-up decision on adding GPU training support).

The server binds to `127.0.0.1` only. `POST /generate` requires an `X-API-Key` header (see `.env.example`), is rate-limited, and rejects oversized request bodies before parsing them — full written plan in `docs/SECURITY.md`. No public exposure without that plan being explicitly reviewed first (CLAUDE.md hard rule 7).

**Image analysis (classical, no ML)** — added outside the numbered phase sequence, by request. Paste an image into the chat UI and get back real, measured properties (dimensions, format, brightness/contrast, dominant colors via median-cut color quantization, EXIF if present) as JSON — deliberately no model, no training, no external API call. See `docs/IMAGE_ANALYSIS.md` and `docs/ROADMAP.md`'s "Features added outside the numbered sequence".

**OCR (from-scratch text extraction)** — same JSON response's `ocr_text` field, built the same way the GPT was: a small CNN character classifier (213,630 params) trained on self-rendered synthetic data, combined with classical connected-component segmentation (Otsu thresholding, flood fill, geometric word/line-break detection) — no pretrained model, no API. Real limitations documented rather than glossed over (case ambiguity between letter pairs like C/c, i/j dot-splitting) — see `docs/OCR.md`.

Checkpoints are now genuinely self-describing: `training/checkpoint.load_for_inference(path)` reconstructs the correct model straight from the checkpoint file, with no separate `model_config.json` needed (Phase 17 found this wasn't actually true before — checkpoints stored the training config, not the architecture — and fixed it). `inference/generate.py`'s CLI no longer takes `--model-config` as a result.

The original tiny 3.8KB placeholder corpus (`data/raw/placeholder_corpus.txt`) clearly overfit in Phase 13 (train loss kept falling, val loss plateaued) — expected at that scale. Phase 16's first experiment (`docs/EXPERIMENTS.md`) tested dataset size as the single changed variable: a ~5.6x larger original corpus (`data/raw/experiment1_larger_corpus.txt`), same architecture, same 300-step budget, and the overfitting gap essentially disappeared (+0.20 → -0.01), with perplexity improving ~5% (11.24 → 10.68).

## Hardware summary (Phase 1)

- CPU: Intel i7-10510U, 4 cores / 8 threads
- RAM: 7.84 GB total (tight — training design targets small models; often <1GB free in practice, watch for this before training runs)
- GPU: NVIDIA GeForce MX330, 2 GB VRAM (not used — PyTorch is installed as a CPU-only build by deliberate choice, see Phase 2)

## Setup

```
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## Usage so far

```
# Rebuild the tokenizer + processed train/val data from data/raw/
.venv\Scripts\python.exe scripts\prepare_data.py

# Run a training smoke test / short training run
.venv\Scripts\python.exe -m unittest discover -s tests

# Generate text from a trained checkpoint
.venv\Scripts\python.exe -m inference.generate --prompt "the model" --temperature 0.8
.venv\Scripts\python.exe -m inference.generate --prompt "the model" --greedy
.venv\Scripts\python.exe -m inference.generate --prompt "the model" --top-k 5
.venv\Scripts\python.exe -m inference.generate --prompt "the model" --top-p 0.9

# Evaluate/compare checkpoints (val loss, perplexity, samples, throughput)
# see training/evaluate.py: evaluate_checkpoint(), format_report(), compare_checkpoints()

# Run the local inference API (binds to 127.0.0.1 only)
# first: copy .env.example to .env and set a real API_KEY (see docs/SECURITY.md)
.venv\Scripts\uvicorn.exe api.main:app --host 127.0.0.1 --port 8000
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json" -H "X-API-Key: <your key>" -d "{\"prompt\": \"the model\"}"

# Classical image analysis (see docs/IMAGE_ANALYSIS.md) -- or paste an image
# into the chat UI's input directly. Response includes ocr_text if the OCR
# checkpoint below has been trained.
curl -X POST http://127.0.0.1:8000/analyze/image -H "X-API-Key: <your key>" -F "file=@photo.png;type=image/png"

# Train the from-scratch OCR character classifier (see docs/OCR.md)
.venv\Scripts\python.exe scripts\train_ocr.py
```

Experiment log for Phase 16 onward: `docs/EXPERIMENTS.md`.

## Project structure

```
llm-project/
├── README.md
├── .gitignore
├── .env.example
├── requirements.txt
├── configs/            # model_config.json, training_config.json
├── data/{raw,processed}/
├── tokenizer/
├── model/              # embeddings.py, attention.py, feedforward.py, transformer.py, gpt.py
├── training/           # dataset.py, train.py, evaluate.py, checkpoint.py
├── inference/          # generate.py
├── analysis/           # image_analysis.py (classical, no ML -- see docs/IMAGE_ANALYSIS.md)
├── ocr/                # from-scratch OCR: synthetic_data.py, model.py, dataset.py, train.py,
│                       # segment.py, normalize.py, extract.py, checkpoint.py -- see docs/OCR.md
├── api/                # main.py, schemas.py
├── tests/
├── scripts/            # inspect_system.py, prepare_data.py, benchmark.py
├── checkpoints/        # gitignored
├── logs/               # gitignored
├── docs/               # ROADMAP.md, phase reports
└── frontend/           # React + Vite app
```

## Rules

Never treat this model as production-equivalent. Legal/local data only. See `CLAUDE.md` for the full rule set.
