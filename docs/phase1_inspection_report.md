# Phase 1 — Server Inspection Report

Date: 2026-09-01

## Hardware / software findings

| Item | Result |
|---|---|
| OS | Windows 11 Home, 64-bit, build 26200 |
| CPU | Intel Core i7-10510U @ 1.8GHz — 4 physical cores / 8 threads |
| RAM | 7.84 GB total, 0.9 GB free at inspection time |
| Disk (C:) | 10.61 GB free — nearly full |
| Disk (D:) | 845 GB free — project lives here |
| GPU | NVIDIA GeForce MX330, 2 GB VRAM, driver 472.19 (CUDA 11.4 capable); Intel UHD Graphics (integrated, not CUDA-capable) |
| CUDA toolkit (nvcc) | Not installed |
| PyTorch | Already installed globally: 2.7.1+cpu (CPU-only build; `torch.cuda.is_available()` is False) |
| Python / pip | 3.11.4 / pip 26.0.1 |
| Git | 2.46.0 |
| Docker | Not installed |
| Network → pypi.org / files.pythonhosted.org | Both reachable on :443 |

## Risk flags
- C: drive is nearly full; Windows pagefile/swap typically lives there. Keep all project data/venv on D:. Monitor RAM before real training runs (Phase 13).
- Only 0.9GB RAM free at inspection time — recheck before any training run and ask the user to close apps if still tight.
- PyTorch is CPU-only despite a physical GPU existing. Whether to install a CUDA build for the 2GB MX330 is a deliberate Phase 2 decision, not a default.

## Model-size proposals

| | Safe | Balanced | Maximum-experimental |
|---|---|---|---|
| Params | ~0.8M | ~11–12M | ~29–30M |
| Vocab size (placeholder, Phase 3 decides) | 100 (char-level) | 2,000 | 8,000 |
| Context length | 128 | 256 | 256 |
| Embedding dim | 128 | 384 | 512 |
| Layers | 4 | 6 | 8 |
| Heads | 4 | 6 | 8 |
| FFN hidden | 512 | 1,536 | 2,048 |
| Batch size | 32 | 16 | 8 |
| LR | 3e-4 | 3e-4 + warmup | 2e-4 + warmup/cosine |
| Precision | fp32 (CPU) | fp32 (CPU) | fp32 (CPU) |
| Est. training RAM | ~0.2–0.5 GB | ~1–1.5 GB | ~2–3 GB+ |

**Recommendation:** Safe — start tiny given 7.84GB total RAM and no usable GPU acceleration yet. Revisit Balanced in Phase 16 (iterative improvement) once the pipeline is verified and free RAM is confirmed healthy. Maximum-experimental is a known ceiling only, not to be run without explicit confirmation and other apps closed.

## Next phase
Phase 2 — Environment setup (venv, `requirements.txt`, `.gitignore`, base `README.md`, config files, git init).
