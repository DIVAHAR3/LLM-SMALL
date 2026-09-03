"""Reproducibility: capturing what it would take to reproduce a given
run. A seed alone isn't enough -- reproducing a result also means
knowing what CODE produced it (git commit), what DATA it trained on
(dataset provenance), and what SOFTWARE/HARDWARE it ran on (library
versions can shift floating-point results even with an identical seed).

set_seed() must be called BEFORE constructing a model: weight
initialization draws from PyTorch's global RNG the moment a layer
(nn.Linear, nn.Conv2d, nn.Embedding, ...) is constructed, so seeding
afterward can no longer make that part reproducible -- only later draws
(DataLoader shuffling, dropout masks) would be affected."""
import json
import os
import platform
import random
import subprocess
from functools import lru_cache
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
DATASET_META_PATH = ROOT / "data" / "processed" / "meta.json"


def set_seed(seed):
    """Seeds every source of randomness this project's training actually
    draws from: Python's random module (used directly by
    training/data_prep.py's document splitting) and PyTorch's RNG
    (weight init, dropout, DataLoader shuffling). torch.manual_seed
    seeds both CPU and any CUDA devices in one call."""
    random.seed(seed)
    torch.manual_seed(seed)


@lru_cache(maxsize=1)
def get_git_commit():
    """Returns (commit_hash, is_dirty). Both None if this isn't a git
    checkout or git isn't available -- reproducibility metadata must
    never crash a real training run over a missing git binary.

    Cached: spawning `git` is a real subprocess (slow on Windows,
    compounded further under antivirus real-time scanning -- measured
    multiple seconds per call on this machine), and the commit can't
    change mid-process anyway. A run that saves many checkpoints (or a
    test suite that calls save_checkpoint many times) would otherwise
    pay that cost on every single save."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=str(ROOT), capture_output=True, text=True, check=True,
        ).stdout
        return commit, bool(status.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None, None


def get_environment_info():
    """Software (and light hardware) context: exact library versions
    can shift floating-point results even given an identical seed, so
    these matter alongside the seed itself, not instead of it."""
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


def get_dataset_info():
    """Reads data/processed/meta.json (written by scripts/prepare_data.py,
    Phase 24) if present -- ties a checkpoint back to exactly which raw
    source file(s) and preprocessing parameters produced its training
    data. Returns None if no processed dataset exists yet."""
    if not DATASET_META_PATH.exists():
        return None
    return json.loads(DATASET_META_PATH.read_text(encoding="utf-8"))


def capture_run_metadata(seed):
    """Bundles everything needed to describe how a run could be
    reproduced. Called automatically by training.checkpoint.save_checkpoint
    -- every checkpoint saved from here on gets this for free."""
    commit, dirty = get_git_commit()
    return {
        "seed": seed,
        "git_commit": commit,
        "git_dirty": dirty,
        "environment": get_environment_info(),
        "dataset": get_dataset_info(),
    }
