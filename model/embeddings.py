import torch
import torch.nn as nn


class GPTEmbedding(nn.Module):
    """Token identity ("what") + learned position ("where") -> one input vector per position.

    Token IDs are arbitrary integers with no inherent numeric meaning, so the
    token embedding table replaces each ID with a dense, learned vector that
    training shapes into a meaningful space. Self-attention (Phase 6) is
    permutation-invariant on its own, so the positional embedding table adds a
    second learned vector per position, giving the model a way to tell "the
    cat sat" apart from "sat the cat".
    """

    def __init__(self, vocab_size, context_length, embedding_dim, dropout=0.0):
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_length, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids):
        batch_size, seq_len = token_ids.shape
        if seq_len > self.context_length:
            raise ValueError(
                f"sequence length {seq_len} exceeds context_length {self.context_length}"
            )
        positions = torch.arange(seq_len, device=token_ids.device)
        token_emb = self.token_embedding(token_ids)    # (B, T, D)
        pos_emb = self.position_embedding(positions)   # (T, D), broadcasts over batch
        return self.dropout(token_emb + pos_emb)        # (B, T, D)
