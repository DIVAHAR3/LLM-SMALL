"""Autoregressive text generation. CLI usage from the project root:

    .venv\\Scripts\\python.exe -m inference.generate --prompt "Hello"
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tokenizer.char_tokenizer import CharTokenizer  # noqa: E402
from training.checkpoint import load_for_inference  # noqa: E402


def _filter_top_k(logits, top_k):
    """Zero out (set to -inf) every logit except the top_k largest."""
    if top_k is None or top_k >= logits.numel():
        return logits
    threshold = torch.topk(logits, top_k).values.min()
    return torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)


def _filter_top_p(logits, top_p):
    """Keep the smallest set of highest-probability tokens whose cumulative
    probability exceeds top_p (nucleus sampling); always keeps at least the
    single most likely token."""
    if top_p is None:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    remove = cumulative_probs > top_p
    remove[1:] = remove[:-1].clone()
    remove[0] = False  # always keep the top token, even alone it may exceed top_p

    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(0, sorted_indices, sorted_logits)
    return filtered


def sample_next_token(logits, temperature=1.0, top_k=None, top_p=None, greedy=False):
    """logits: 1D tensor over the vocabulary for a single next-token prediction."""
    if greedy:
        return int(torch.argmax(logits).item())

    scaled = logits / temperature
    scaled = _filter_top_k(scaled, top_k)
    scaled = _filter_top_p(scaled, top_p)
    probs = torch.softmax(scaled, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def generate_ids(model, prompt_ids, max_new_tokens, temperature=1.0, top_k=None, top_p=None, greedy=False, stop_token_ids=None):
    """Core autoregressive loop, operating on token ids directly (no
    tokenizer needed here -- keeps this the most directly testable unit).
    Feeds only the most recent context_length tokens to the model at each
    step, since the model has no way to attend further back than that."""
    context_length = model.config["context_length"]
    stop_token_ids = set(stop_token_ids) if stop_token_ids else set()

    was_training = model.training
    model.eval()
    ids = list(prompt_ids)
    try:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                context = ids[-context_length:]
                x = torch.tensor([context], dtype=torch.long)
                logits = model(x)[0, -1]
                next_id = sample_next_token(logits, temperature, top_k, top_p, greedy)
                ids.append(next_id)
                if next_id in stop_token_ids:
                    break
    finally:
        model.train(was_training)
    return ids


def generate_text(model, tokenizer, prompt, max_new_tokens=200, temperature=1.0, top_k=None, top_p=None, greedy=False, stop_token_ids=None):
    prompt_ids = tokenizer.encode(prompt)
    ids = generate_ids(model, prompt_ids, max_new_tokens, temperature, top_k, top_p, greedy, stop_token_ids)
    return tokenizer.decode(ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "phase13_run.pt"))
    parser.add_argument("--tokenizer-path", default=str(ROOT / "tokenizer" / "vocab.json"))
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    tokenizer = CharTokenizer.load(args.tokenizer_path)
    # Model architecture comes entirely from the checkpoint itself (Phase 17:
    # checkpoints are self-describing) -- no separate model config needed.
    model, _ = load_for_inference(args.checkpoint)

    text = generate_text(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        top_k=args.top_k, top_p=args.top_p, greedy=args.greedy,
    )
    print(text)


if __name__ == "__main__":
    main()
