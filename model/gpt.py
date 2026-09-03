import torch
import torch.nn as nn

from model.embeddings import GPTEmbedding
from model.normalization import LayerNorm
from model.transformer import TransformerBlock

_CONFIG_KEYS = {
    "vocab_size",
    "context_length",
    "embedding_dim",
    "num_layers",
    "num_heads",
    "ffn_hidden_dim",
    "dropout",
    "qkv_bias",
}


class GPTModel(nn.Module):
    """Full decoder-only GPT: embeddings -> N transformer blocks -> final
    LayerNorm -> LM head, producing next-token logits of shape
    (batch, seq_len, vocab_size).

    Weight tying between the token embedding and the LM head is a common
    optimization (fewer params, often better quality) but is explicitly a
    Phase 35 topic to compare deliberately -- kept as two separate
    matrices here rather than skipping ahead.
    """

    def __init__(
        self,
        vocab_size,
        context_length,
        embedding_dim,
        num_layers,
        num_heads,
        ffn_hidden_dim,
        dropout=0.0,
        qkv_bias=False,
    ):
        super().__init__()
        self.config = {
            "vocab_size": vocab_size,
            "context_length": context_length,
            "embedding_dim": embedding_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "ffn_hidden_dim": ffn_hidden_dim,
            "dropout": dropout,
            "qkv_bias": qkv_bias,
        }

        self.embedding = GPTEmbedding(vocab_size, context_length, embedding_dim, dropout=dropout)
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(embedding_dim, num_heads, ffn_hidden_dim, dropout=dropout, qkv_bias=qkv_bias)
                for _ in range(num_layers)
            ]
        )
        self.final_ln = LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)

    @classmethod
    def from_config(cls, config):
        """Build a model straight from a model_config.json-shaped dict,
        ignoring any non-constructor keys (e.g. "_comment")."""
        kwargs = {k: v for k, v in config.items() if k in _CONFIG_KEYS}
        return cls(**kwargs)

    def forward(self, token_ids):
        x = self.embedding(token_ids)
        x = self.blocks(x)
        x = self.final_ln(x)
        return self.lm_head(x)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def resize_vocab(self, new_vocab_size):
        """Grows the token embedding and LM head to new_vocab_size,
        in place. The first old_vocab_size rows are copied EXACTLY from
        the current (possibly pretrained) weights; only the new rows get
        fresh random initialization. New token ids must be appended at
        the end of the tokenizer's vocabulary (see
        CharTokenizer.with_additional_special_tokens) for this to make
        sense -- every existing id must keep pointing at the same
        embedding row it was actually trained on."""
        old_vocab_size = self.config["vocab_size"]
        if new_vocab_size < old_vocab_size:
            raise ValueError(f"new_vocab_size ({new_vocab_size}) must be >= current vocab_size ({old_vocab_size})")
        if new_vocab_size == old_vocab_size:
            return self

        embedding_dim = self.config["embedding_dim"]
        old_token_embedding = self.embedding.token_embedding
        old_lm_head = self.lm_head

        new_token_embedding = nn.Embedding(new_vocab_size, embedding_dim)
        new_lm_head = nn.Linear(embedding_dim, new_vocab_size, bias=False)
        with torch.no_grad():
            new_token_embedding.weight[:old_vocab_size] = old_token_embedding.weight
            new_lm_head.weight[:old_vocab_size] = old_lm_head.weight

        self.embedding.token_embedding = new_token_embedding
        self.lm_head = new_lm_head
        self.config["vocab_size"] = new_vocab_size
        return self

    def summary(self):
        config_str = ", ".join(f"{k}={v}" for k, v in self.config.items())
        lines = [f"GPTModel({config_str})"]
        for name, module in self.named_children():
            n = sum(p.numel() for p in module.parameters())
            lines.append(f"  {name:<12} {n:>10,} params")
        lines.append(f"  {'TOTAL':<12} {self.num_parameters():>10,} params")
        return "\n".join(lines)
