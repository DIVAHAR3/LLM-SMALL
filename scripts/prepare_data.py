"""Raw -> paragraphs -> clean -> filter malformed -> dedupe -> reproducible
document-level split -> tokenize pipeline (Phase 24 hardened). Accepts one
or more raw files; run from the project root:

    .venv\\Scripts\\python.exe scripts\\prepare_data.py
    .venv\\Scripts\\python.exe scripts\\prepare_data.py --raw-file a.txt b.txt
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.data_prep import prepare_corpus  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-file", nargs="+", default=[str(ROOT / "data" / "raw" / "placeholder_corpus.txt")])
    parser.add_argument("--processed-dir", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--tokenizer-path", default=str(ROOT / "tokenizer" / "vocab.json"))
    parser.add_argument("--model-config", default=str(ROOT / "configs" / "model_config.json"))
    parser.add_argument("--training-config", default=str(ROOT / "configs" / "training_config.json"))
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--min-doc-chars", type=int, default=20)
    args = parser.parse_args()

    training_config = json.loads(Path(args.training_config).read_text(encoding="utf-8"))
    val_split_ratio = training_config["val_split_ratio"]

    raw_texts = [Path(p).read_text(encoding="utf-8") for p in args.raw_file]

    train_ids, val_ids, tokenizer, stats = prepare_corpus(
        raw_texts, val_split_ratio, seed=args.seed, min_doc_chars=args.min_doc_chars,
    )

    Path(args.tokenizer_path).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(args.tokenizer_path)

    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "train_ids.json").write_text(json.dumps(train_ids), encoding="utf-8")
    (processed_dir / "val_ids.json").write_text(json.dumps(val_ids), encoding="utf-8")

    meta = {
        "source_files": [str(p) for p in args.raw_file],
        "val_split_ratio": val_split_ratio,
        "seed": args.seed,
        "min_doc_chars": args.min_doc_chars,
        **stats,
    }
    (processed_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    model_config_path = Path(args.model_config)
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    old_vocab_size = model_config["vocab_size"]
    model_config["vocab_size"] = tokenizer.vocab_size
    model_config_path.write_text(json.dumps(model_config, indent=2), encoding="utf-8")

    print(json.dumps(meta, indent=2))
    if old_vocab_size != tokenizer.vocab_size:
        print(f"\nUpdated {args.model_config} vocab_size: {old_vocab_size} -> {tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
