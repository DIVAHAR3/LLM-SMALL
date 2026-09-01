import torch.nn as nn

from model.attention import MultiHeadAttention
from model.feedforward import FeedForward
from model.normalization import LayerNorm
from model.residual import ResidualConnection


class TransformerBlock(nn.Module):
    """One decoder block: Pre-LN residual attention, then Pre-LN residual FFN.

        x = x + MHSA(LN(x))
        x = x + FFN(LN(x))

    Pre-LN (normalizing BEFORE the sublayer, rather than after as the
    original 2017 Transformer paper did) trains more stably without careful
    learning-rate warmup, which matters given how little tuning budget this
    hardware allows. Reuses ResidualConnection from Phase 8 directly: each
    sublayer here is just "LayerNorm then MHSA/FFN" wrapped in the same
    residual pattern already built and tested.
    """

    def __init__(self, embedding_dim, num_heads, ffn_hidden_dim, dropout=0.0, qkv_bias=False):
        super().__init__()
        self.attn_block = ResidualConnection(
            nn.Sequential(
                LayerNorm(embedding_dim),
                MultiHeadAttention(embedding_dim, num_heads, causal=True, qkv_bias=qkv_bias, dropout=dropout),
            )
        )
        self.ffn_block = ResidualConnection(
            nn.Sequential(
                LayerNorm(embedding_dim),
                FeedForward(embedding_dim, ffn_hidden_dim, dropout=dropout),
            )
        )

    def forward(self, x):
        x = self.attn_block(x)
        x = self.ffn_block(x)
        return x
