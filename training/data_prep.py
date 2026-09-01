import re

from tokenizer.char_tokenizer import CharTokenizer


def clean_text(text):
    """Normalize line endings, strip control characters, collapse excess blank lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ch.isprintable())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_tokenizer(text):
    return CharTokenizer.from_text(text)


def split_ids(ids, val_split_ratio):
    """Contiguous split (not shuffled) — shuffling would break the sequential
    structure a language model needs and would leak adjacent context across
    the train/val boundary."""
    if not 0 < val_split_ratio < 1:
        raise ValueError(f"val_split_ratio must be in (0, 1), got {val_split_ratio}")
    split_at = int(len(ids) * (1 - val_split_ratio))
    return ids[:split_at], ids[split_at:]
