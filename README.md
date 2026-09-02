# GPT-from-Scratch

An educational, decoder-only GPT-style Transformer built in PyTorch from scratch, trained on this machine, and eventually served via FastAPI with a React/Vite chat UI.

This is a learning project, not a production system — see `CLAUDE.md` for the governing rules and `docs/ROADMAP.md` for the full phase-by-phase plan.

## Status

Completed through **Phase 19 — API security**. Full decoder-only GPT (embeddings → 4 transformer blocks → final LayerNorm → LM head, 821,248 params, Safe tier) trains end-to-end on CPU, checkpoints/resumes correctly, generates text via `python -m inference.generate --prompt "..."` (greedy, temperature, top-k, top-p, stop tokens all supported), can be evaluated/compared via `training/evaluate.py` (val loss, perplexity, sample generations, throughput), and is served locally over HTTP via FastAPI (`GET /health`, `POST /generate`) — see Usage below. See `docs/phase1_inspection_report.md` for the original hardware findings and model-size tiers.

The server binds to `127.0.0.1` only. `POST /generate` requires an `X-API-Key` header (see `.env.example`), is rate-limited, and rejects oversized request bodies before parsing them — full written plan in `docs/SECURITY.md`. No public exposure without that plan being explicitly reviewed first (CLAUDE.md hard rule 7).

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
