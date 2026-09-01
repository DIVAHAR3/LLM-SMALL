# GPT-from-Scratch

An educational, decoder-only GPT-style Transformer built in PyTorch from scratch, trained on this machine, and eventually served via FastAPI with a React/Vite chat UI.

This is a learning project, not a production system — see `CLAUDE.md` for the governing rules and `docs/ROADMAP.md` for the full phase-by-phase plan.

## Status

Currently on **Phase 2 — Environment setup**. See `docs/phase1_inspection_report.md` for hardware findings and the recommended model-size tier.

## Hardware summary (Phase 1)

- CPU: Intel i7-10510U, 4 cores / 8 threads
- RAM: 7.84 GB total (tight — training design targets small models)
- GPU: NVIDIA GeForce MX330, 2 GB VRAM (CUDA availability TBD in Phase 2)
- PyTorch build: TBD this phase (CPU vs CUDA — see phase 2 report)

## Setup

```
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

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
