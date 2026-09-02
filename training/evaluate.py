import math
import time
import tracemalloc
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from inference.generate import generate_text
from model.gpt import GPTModel
from training.checkpoint import load_checkpoint
from training.dataset import TextDataset
from training.loss import cross_entropy_loss


def perplexity(loss):
    """exp(cross-entropy loss): "the model is as confused as if it were
    guessing uniformly among this many equally likely next tokens." A
    freshly-initialized model's perplexity sits near vocab_size (see
    Phase 11's log(vocab_size) baseline); a trained model should be well
    below that."""
    return math.exp(loss)


def evaluate_checkpoint(checkpoint_path, model_config, tokenizer, val_ids, batch_size=32, sample_prompts=None, max_new_tokens=80):
    """Loads a checkpoint fresh (so multiple checkpoints can be compared
    independently) and reports val loss/perplexity, throughput, and a few
    qualitative sample generations, plus a best-effort in-process memory
    figure (see peak_python_object_memory_mb caveat below).

    Memory caveat: tracemalloc only tracks allocations made through
    Python's own memory allocator. PyTorch tensor storage is allocated
    through its own C++ allocator instead, so tracemalloc systematically
    misses almost all of a model's actual memory footprint -- the reported
    number reflects small Python-level object overhead only, not real
    tensor memory. There is no dependency-free, cross-platform way to read
    true process RSS (psutil would add a new dependency for this alone),
    so real memory monitoring for this project continues to rely on the
    OS-level free-RAM before/after checks done manually around training
    runs (as in Phase 13), not this in-process figure.
    """
    tracemalloc.start()  # started before model construction, so it captures the full call, not just the eval loop
    model = GPTModel.from_config(model_config)
    checkpoint = load_checkpoint(checkpoint_path, model)
    model.eval()

    val_loader = DataLoader(
        TextDataset(val_ids, model_config["context_length"]), batch_size=batch_size, shuffle=False
    )

    t0 = time.time()
    total_loss, tokens_processed = 0.0, 0
    with torch.no_grad():
        for x, y in val_loader:
            loss = cross_entropy_loss(model(x), y)
            total_loss += loss.item() * y.numel()
            tokens_processed += y.numel()
    elapsed = time.time() - t0  # scoped to just the loss-evaluation loop, for a clean throughput figure

    avg_val_loss = total_loss / tokens_processed
    metrics = checkpoint.get("metrics", {}) or {}

    samples = []
    if sample_prompts:
        for prompt in sample_prompts:
            text = generate_text(model, tokenizer, prompt, max_new_tokens=max_new_tokens, temperature=0.8)
            samples.append({"prompt": prompt, "generated": text})

    # Memory window spans the whole call (model construction through
    # generation), unlike eval_seconds -- see the memory caveat above.
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "checkpoint_path": str(checkpoint_path),
        "epoch": checkpoint.get("epoch"),
        "step": checkpoint.get("step"),
        "param_count": model.num_parameters(),
        "val_loss": avg_val_loss,
        "perplexity": perplexity(avg_val_loss),
        "tokens_processed": tokens_processed,
        "eval_seconds": elapsed,
        "tokens_per_sec": tokens_processed / elapsed if elapsed > 0 else float("inf"),
        "peak_python_object_memory_mb": peak_memory / (1024 * 1024),
        "train_loss_history_tail": metrics.get("train_loss", [])[-5:],
        "val_loss_history_tail": metrics.get("val_loss", [])[-5:],
        "samples": samples,
    }


def format_report(report):
    lines = [
        f"Checkpoint: {report['checkpoint_path']}",
        f"  epoch={report['epoch']}  step={report['step']}  params={report['param_count']:,}",
        f"  val_loss={report['val_loss']:.4f}  perplexity={report['perplexity']:.2f}",
        f"  tokens_processed={report['tokens_processed']:,}  "
        f"eval_time={report['eval_seconds']:.2f}s  "
        f"throughput={report['tokens_per_sec']:,.0f} tok/s",
        f"  peak Python object memory: {report['peak_python_object_memory_mb']:.4f} MB "
        f"(excludes PyTorch tensor storage -- not a real total; see docstring)",
    ]
    if report["train_loss_history_tail"]:
        lines.append(f"  train_loss (last 5 logged steps): {[round(v, 4) for v in report['train_loss_history_tail']]}")
    if report["val_loss_history_tail"]:
        lines.append(f"  val_loss (last 5 logged evals): {[round(v, 4) for v in report['val_loss_history_tail']]}")
    for s in report["samples"]:
        lines.append(f"  sample (prompt={s['prompt']!r}):")
        lines.append(f"    {s['generated']}")
    return "\n".join(lines)


def compare_checkpoints(reports):
    header = f"{'checkpoint':<28} {'step':>6} {'val_loss':>10} {'perplexity':>12} {'params':>10}"
    lines = ["Checkpoint comparison:", header]
    for r in reports:
        name = Path(r["checkpoint_path"]).name
        lines.append(f"{name:<28} {r['step']:>6} {r['val_loss']:>10.4f} {r['perplexity']:>12.2f} {r['param_count']:>10,}")
    return "\n".join(lines)
