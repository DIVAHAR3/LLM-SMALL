import json
import re
from collections import Counter
from pathlib import Path

# Splits text into alternating whitespace-run and non-whitespace-run chunks,
# covering every character with no loss -- BPE merges happen WITHIN each
# chunk only (never across chunk boundaries), and decoding is a simple
# concatenation, so this pre-tokenization keeps encode/decode exactly
# lossless without needing special whitespace handling.
_PRETOKENIZE_PATTERN = re.compile(r"\s+|\S+")


class BPETokenizer:
    """Byte-pair-encoding subword tokenizer, learned from a corpus.

    Vocabulary = special tokens + base characters + merged tokens, in the
    order they were learned. Encoding replays the same merges in that same
    priority order, so an input never seen during training still encodes
    correctly -- it just falls back to smaller, more numerous pieces.
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

    def __init__(self, token_to_id, merges):
        self.token_to_id = token_to_id
        self.id_to_token = {i: t for t, i in token_to_id.items()}
        self.merges = merges  # ordered list of (token_a, token_b) tuples, learned-priority order
        self.merge_rank = {pair: rank for rank, pair in enumerate(merges)}

    @classmethod
    def from_text(cls, text, vocab_size):
        """Learns merges greedily: at each step, merge whichever adjacent
        token pair is currently most frequent across the whole corpus,
        until vocab_size is reached or no mergeable pairs remain."""
        min_vocab_size = len(cls.SPECIAL_TOKENS) + 1
        if vocab_size < min_vocab_size:
            raise ValueError(f"vocab_size must be >= {min_vocab_size}, got {vocab_size}")

        chunks = _PRETOKENIZE_PATTERN.findall(text)
        word_freqs = Counter(tuple(chunk) for chunk in chunks)

        base_chars = sorted({ch for word in word_freqs for ch in word})
        vocab_list = list(cls.SPECIAL_TOKENS) + base_chars
        merges = []

        num_merges_needed = max(0, vocab_size - len(vocab_list))
        for _ in range(num_merges_needed):
            pair_counts = Counter()
            for word, freq in word_freqs.items():
                for i in range(len(word) - 1):
                    pair_counts[(word[i], word[i + 1])] += freq
            if not pair_counts:
                break  # every word is already a single token; nothing left to merge

            best_pair = max(pair_counts.items(), key=lambda kv: kv[1])[0]
            merged_token = best_pair[0] + best_pair[1]

            new_word_freqs = Counter()
            for word, freq in word_freqs.items():
                new_word_freqs[tuple(_merge_all_occurrences(word, best_pair, merged_token))] += freq
            word_freqs = new_word_freqs

            vocab_list.append(merged_token)
            merges.append(best_pair)

        token_to_id = {tok: i for i, tok in enumerate(vocab_list)}
        return cls(token_to_id, merges)

    def _bpe_encode_word(self, word_chars):
        word = list(word_chars)
        while len(word) > 1:
            pairs_present = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
            candidates = [p for p in pairs_present if p in self.merge_rank]
            if not candidates:
                break
            # lowest rank = learned earliest = highest merge priority
            best_pair = min(candidates, key=lambda p: self.merge_rank[p])
            merged_token = best_pair[0] + best_pair[1]
            word = _merge_all_occurrences(word, best_pair, merged_token)
        return word

    def encode(self, text, add_special_tokens=False):
        chunks = _PRETOKENIZE_PATTERN.findall(text)
        unk_id = self.token_to_id[self.UNK_TOKEN]
        ids = []
        for chunk in chunks:
            for piece in self._bpe_encode_word(tuple(chunk)):
                ids.append(self.token_to_id.get(piece, unk_id))
        if add_special_tokens:
            ids = [self.token_to_id[self.BOS_TOKEN]] + ids + [self.token_to_id[self.EOS_TOKEN]]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        special_ids = {self.token_to_id[t] for t in self.SPECIAL_TOKENS}
        pieces = []
        for i in ids:
            if skip_special_tokens and i in special_ids:
                continue
            pieces.append(self.id_to_token.get(i, self.UNK_TOKEN))
        return "".join(pieces)

    @property
    def vocab_size(self):
        return len(self.token_to_id)

    def save(self, path):
        data = {
            "token_to_id": self.token_to_id,
            "merges": [list(pair) for pair in self.merges],
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        merges = [tuple(pair) for pair in data["merges"]]
        return cls(data["token_to_id"], merges)


def _merge_all_occurrences(word, pair, merged_token):
    """Merges EVERY adjacent occurrence of `pair` in `word` in one
    left-to-right pass -- both training (counting frequencies) and
    encoding (replaying learned merges) need this exact semantics, not
    just the first occurrence, to stay consistent with each other."""
    new_word = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
            new_word.append(merged_token)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return new_word
