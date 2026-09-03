"""Benchmarking: tokens/sec, latency, first-token latency, load time, and
available system memory, for both real inference (the trained checkpoint)
and raw architecture compute cost across batch sizes and context lengths.

No GPU comparison -- this project runs CPU-only by deliberate choice
(Phase 2). This script only measures; it makes no optimization changes
(that's Phase 23+).

Run from the project root:
    .venv\\Scripts\\python.exe scripts\\benchmark.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from inference.generate import stream_ids  # noqa: E402
from model.gpt import GPTModel  # noqa: E402
from tokenizer.char_tokenizer import CharTokenizer  # noqa: E402
from training.checkpoint import load_for_inference  # noqa: E402
from training.loss import cross_entropy_loss  # noqa: E402
from training.monitoring import get_available_memory_mb  # noqa: E402


def benchmark_inference(checkpoint_path, tokenizer_path, prompt, max_new_tokens_list, repeats=3):
    """Real inference performance: actual trained checkpoint, actual
    tokenizer, actual autoregressive generation via stream_ids -- the same
    code path the API and CLI use."""
    mem_before_load = get_available_memory_mb()
    t0 = time.perf_counter()
    tokenizer = CharTokenizer.load(tokenizer_path)
    model, _ = load_for_inference(checkpoint_path)
    load_time = time.perf_counter() - t0
    mem_after_load = get_available_memory_mb()

    prompt_ids = tokenizer.encode(prompt)
    configs = []

    for max_new_tokens in max_new_tokens_list:
        first_token_times, total_times = [], []
        for _ in range(repeats):
            t_start = time.perf_counter()
            gen = stream_ids(model, prompt_ids, max_new_tokens, greedy=True)
            next(gen)
            t_first = time.perf_counter()
            list(gen)  # drain the rest
            t_end = time.perf_counter()
            first_token_times.append(t_first - t_start)
            total_times.append(t_end - t_start)

        avg_first_token = sum(first_token_times) / repeats
        avg_total = sum(total_times) / repeats
        steady_state_tokens = max_new_tokens - 1
        steady_state_time = avg_total - avg_first_token
        tokens_per_sec = steady_state_tokens / steady_state_time if steady_state_time > 0 else float("inf")

        configs.append({
            "max_new_tokens": max_new_tokens,
            "avg_first_token_latency_ms": avg_first_token * 1000,
            "avg_total_latency_ms": avg_total * 1000,
            "steady_state_tokens_per_sec": tokens_per_sec,
        })

    return {
        "checkpoint": str(checkpoint_path),
        "param_count": model.num_parameters(),
        "load_time_seconds": load_time,
        "available_memory_before_load_mb": mem_before_load,
        "available_memory_after_load_mb": mem_after_load,
        "configs": configs,
    }


def benchmark_compute_cost(batch_size, context_length, model_config, repeats=5, warmup=2):
    """Raw forward+backward+optimizer-step cost for a given (batch_size,
    context_length), using a freshly-initialized (untrained) model -- this
    measures compute cost, not output quality, so no trained checkpoint is
    needed to compare configurations that were never actually trained.
    Every architecture dimension besides context_length (the axis being
    swept) comes from model_config -- the project's single source of
    truth (configs/model_config.json) -- rather than being duplicated
    here as separate literals that could silently drift out of sync."""
    torch.manual_seed(0)
    config = {**model_config, "context_length": context_length}
    model = GPTModel.from_config(config)
    vocab_size = config["vocab_size"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.randint(0, vocab_size, (batch_size, context_length))
    y = torch.randint(0, vocab_size, (batch_size, context_length))

    def step():
        logits = model(x)
        loss = cross_entropy_loss(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for _ in range(warmup):
        step()

    t0 = time.perf_counter()
    for _ in range(repeats):
        step()
    elapsed = time.perf_counter() - t0

    step_time = elapsed / repeats
    tokens_per_step = batch_size * context_length
    return {
        "batch_size": batch_size,
        "context_length": context_length,
        "param_count": model.num_parameters(),
        "avg_step_time_ms": step_time * 1000,
        "tokens_per_sec": tokens_per_step / step_time,
    }


def format_inference_report(report):
    lines = [
        f"Checkpoint: {report['checkpoint']}",
        f"  params: {report['param_count']:,}",
        f"  load_time: {report['load_time_seconds']:.3f}s",
    ]
    if report["available_memory_before_load_mb"] is not None:
        lines.append(
            f"  available memory: {report['available_memory_before_load_mb']:.0f} MB before load -> "
            f"{report['available_memory_after_load_mb']:.0f} MB after"
        )
    else:
        lines.append("  available memory: unavailable (non-Windows or query failed)")
    lines.append(f"  {'max_new_tokens':>15} {'first_token_ms':>16} {'total_ms':>10} {'tokens/sec':>12}")
    for c in report["configs"]:
        lines.append(
            f"  {c['max_new_tokens']:>15} {c['avg_first_token_latency_ms']:>16.1f} "
            f"{c['avg_total_latency_ms']:>10.1f} {c['steady_state_tokens_per_sec']:>12.1f}"
        )
    return "\n".join(lines)


def format_compute_cost_report(results):
    lines = [f"{'batch_size':>10} {'context_length':>14} {'params':>10} {'step_ms':>10} {'tokens/sec':>12}"]
    for r in results:
        lines.append(
            f"{r['batch_size']:>10} {r['context_length']:>14} {r['param_count']:>10,} "
            f"{r['avg_step_time_ms']:>10.1f} {r['tokens_per_sec']:>12.1f}"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "phase13_run.pt"))
    parser.add_argument("--tokenizer-path", default=str(ROOT / "tokenizer" / "vocab.json"))
    parser.add_argument("--prompt", default="the model")
    parser.add_argument("--output", default=str(ROOT / "docs" / "benchmark_results.json"))
    parser.add_argument("--model-config", default=str(ROOT / "configs" / "model_config.json"))
    args = parser.parse_args()

    model_config = json.loads(Path(args.model_config).read_text(encoding="utf-8"))

    print("=== Inference benchmark (real trained checkpoint) ===")
    inference_report = benchmark_inference(
        args.checkpoint, args.tokenizer_path, args.prompt, max_new_tokens_list=[20, 100, 300], repeats=3,
    )
    print(format_inference_report(inference_report))

    print("\n=== Compute-cost benchmark (fresh untrained models, batch_size x context_length) ===")
    configs = [
        (8, 64), (32, 64),
        (8, 128), (32, 128),
        (32, 256),
    ]
    compute_results = [benchmark_compute_cost(b, c, model_config) for b, c in configs]
    print(format_compute_cost_report(compute_results))

    output = {"inference": inference_report, "compute_cost": compute_results}
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved full results to {args.output}")


if __name__ == "__main__":
    main()
