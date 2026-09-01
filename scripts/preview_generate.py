"""Informal preview of the current checkpoint's output -- NOT the formal
Phase 14 generation module (that will add proper sampling strategies,
stop tokens, and a real CLI). Run from the project root:

    .venv\\Scripts\\python.exe scripts\\preview_generate.py --prompt "Hello"
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.gpt import GPTModel  # noqa: E402
from tokenizer.char_tokenizer import CharTokenizer  # noqa: E402
from training.checkpoint import load_checkpoint  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="A tiny language model")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "phase13_run.pt"))
    parser.add_argument("--model-config", default=str(ROOT / "configs" / "model_config.json"))
    parser.add_argument("--tokenizer-path", default=str(ROOT / "tokenizer" / "vocab.json"))
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    model_cfg = json.loads(Path(args.model_config).read_text(encoding="utf-8"))
    tokenizer = CharTokenizer.load(args.tokenizer_path)
    model = GPTModel.from_config(model_cfg)
    load_checkpoint(args.checkpoint, model)
    model.eval()

    context_length = model_cfg["context_length"]
    ids = tokenizer.encode(args.prompt)

    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            context = ids[-context_length:]
            x = torch.tensor([context], dtype=torch.long)
            logits = model(x)
            next_logits = logits[0, -1] / args.temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            ids.append(next_id)

    print(tokenizer.decode(ids))


if __name__ == "__main__":
    main()
