import json
from pathlib import Path


class CharTokenizer:
    """Character-level tokenizer with a fixed set of reserved special tokens."""

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

    def __init__(self, char_to_id):
        self.char_to_id = char_to_id
        self.id_to_char = {i: ch for ch, i in char_to_id.items()}

    @classmethod
    def from_text(cls, text):
        chars = sorted(set(text))
        vocab = cls.SPECIAL_TOKENS + chars
        char_to_id = {ch: i for i, ch in enumerate(vocab)}
        return cls(char_to_id)

    @property
    def vocab_size(self):
        return len(self.char_to_id)

    def encode(self, text, add_special_tokens=False):
        unk_id = self.char_to_id[self.UNK_TOKEN]
        ids = [self.char_to_id.get(ch, unk_id) for ch in text]
        if add_special_tokens:
            ids = [self.char_to_id[self.BOS_TOKEN]] + ids + [self.char_to_id[self.EOS_TOKEN]]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        special_ids = {self.char_to_id[t] for t in self.SPECIAL_TOKENS}
        chars = []
        for i in ids:
            if skip_special_tokens and i in special_ids:
                continue
            chars.append(self.id_to_char.get(i, self.UNK_TOKEN))
        return "".join(chars)

    def save(self, path):
        Path(path).write_text(
            json.dumps({"char_to_id": self.char_to_id}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["char_to_id"])
