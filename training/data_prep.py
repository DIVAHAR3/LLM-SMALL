import random
import re
import unicodedata

from tokenizer.char_tokenizer import CharTokenizer


def clean_text(text):
    """Normalize line endings and Unicode form, strip control characters,
    collapse excess blank lines. NFC normalization matters because a
    character-level tokenizer treats different Unicode representations of
    the "same" visible character as different vocabulary entries -- without
    it, a corpus mixing composed and decomposed forms would silently
    fragment its own vocabulary and statistics."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
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


def split_into_paragraphs(raw_text):
    """Splits raw text into paragraph-level "documents" on blank-line
    boundaries -- the pipeline's document granularity: fine enough to catch
    genuine duplicate content between source files, coarse enough to keep
    each unit's own internal sequential structure intact."""
    normalized = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = re.split(r"\n\s*\n", normalized)
    return [p.strip() for p in paragraphs if p.strip()]


def deduplicate_documents(documents):
    """Exact-match dedup, preserving first-occurrence order. Returns
    (deduped_documents, num_duplicates_removed)."""
    seen = set()
    deduped = []
    for doc in documents:
        if doc in seen:
            continue
        seen.add(doc)
        deduped.append(doc)
    return deduped, len(documents) - len(deduped)


def filter_malformed_documents(documents, min_chars=20):
    """Drops documents that are empty, whitespace-only (already handled by
    split_into_paragraphs), or too short to carry meaningful sequential
    structure once cleaned -- stray fragments, headers, etc. Returns
    (kept_documents, num_removed)."""
    kept = [doc for doc in documents if len(doc) >= min_chars]
    return kept, len(documents) - len(kept)


def split_documents(documents, val_split_ratio, seed=1337):
    """Reproducible, seeded document-level split: shuffles a COPY of the
    document list (never a corpus's raw character stream) so the same seed
    always produces the same train/val assignment, and no single document's
    content is ever split across train and val."""
    if not 0 < val_split_ratio < 1:
        raise ValueError(f"val_split_ratio must be in (0, 1), got {val_split_ratio}")
    shuffled = list(documents)
    random.Random(seed).shuffle(shuffled)
    split_at = int(len(shuffled) * (1 - val_split_ratio))
    return shuffled[:split_at], shuffled[split_at:]


def compute_dataset_stats(documents, tokenizer, train_ids, val_ids, duplicates_removed=0, malformed_removed=0):
    doc_lengths = [len(doc) for doc in documents]
    return {
        "num_documents": len(documents),
        "duplicates_removed": duplicates_removed,
        "malformed_removed": malformed_removed,
        "total_chars": sum(doc_lengths),
        "avg_doc_length_chars": sum(doc_lengths) / len(documents) if documents else 0,
        "min_doc_length_chars": min(doc_lengths) if documents else 0,
        "max_doc_length_chars": max(doc_lengths) if documents else 0,
        "vocab_size": tokenizer.vocab_size,
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
        "total_tokens": len(train_ids) + len(val_ids),
    }


def prepare_corpus(raw_texts, val_split_ratio, seed=1337, min_doc_chars=20):
    """Full hardened pipeline: one or more raw texts -> paragraph documents
    -> cleaned -> malformed-filtered -> deduplicated -> reproducibly split
    at the document level -> tokenized. Returns (train_ids, val_ids,
    tokenizer, stats)."""
    documents = []
    for raw_text in raw_texts:
        documents.extend(clean_text(p) for p in split_into_paragraphs(raw_text))

    documents, malformed_removed = filter_malformed_documents(documents, min_chars=min_doc_chars)
    documents, duplicates_removed = deduplicate_documents(documents)

    train_docs, val_docs = split_documents(documents, val_split_ratio, seed=seed)

    tokenizer = build_tokenizer("\n\n".join(documents))
    train_ids = tokenizer.encode("\n\n".join(train_docs))
    val_ids = tokenizer.encode("\n\n".join(val_docs))

    stats = compute_dataset_stats(
        documents, tokenizer, train_ids, val_ids,
        duplicates_removed=duplicates_removed, malformed_removed=malformed_removed,
    )
    return train_ids, val_ids, tokenizer, stats
