import math

import torch
import torch.nn as nn


def scaled_dot_product_attention(q, k, v, causal=False):
    """Core attention primitive, shared by single-head and multi-head attention.

    q: (..., T_q, d_k)   k: (..., T_k, d_k)   v: (..., T_k, d_v)
    Returns (output, attn_weights) where output is (..., T_q, d_v) and
    attn_weights is (..., T_q, T_k), each row summing to 1.

    Leading "..." dims broadcast identically for q/k/v, so this same function
    works for a single head (no extra dim) or multiple heads (an extra head
    dim before T_q/T_k) without any special-casing.
    """
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)  # (..., T_q, T_k)

    if causal:
        T_q, T_k = scores.size(-2), scores.size(-1)
        future_mask = torch.triu(
            torch.ones(T_q, T_k, dtype=torch.bool, device=scores.device), diagonal=1
        )
        scores = scores.masked_fill(future_mask, float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)
    output = attn_weights @ v
    return output, attn_weights


class SingleHeadAttention(nn.Module):
    """One attention head: learns its own Q/K/V projections down to head_dim."""

    def __init__(self, embedding_dim, head_dim, causal=True, qkv_bias=False):
        super().__init__()
        self.causal = causal
        self.W_q = nn.Linear(embedding_dim, head_dim, bias=qkv_bias)
        self.W_k = nn.Linear(embedding_dim, head_dim, bias=qkv_bias)
        self.W_v = nn.Linear(embedding_dim, head_dim, bias=qkv_bias)
        self.last_attn_weights = None

    def forward(self, x):
        q, k, v = self.W_q(x), self.W_k(x), self.W_v(x)
        out, attn_weights = scaled_dot_product_attention(q, k, v, causal=self.causal)
        self.last_attn_weights = attn_weights
        return out


class MultiHeadAttention(nn.Module):
    """Splits embedding_dim into num_heads parallel attention heads, then
    concatenates and projects back to embedding_dim."""

    def __init__(self, embedding_dim, num_heads, causal=True, qkv_bias=False, dropout=0.0):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.causal = causal

        self.W_q = nn.Linear(embedding_dim, embedding_dim, bias=qkv_bias)
        self.W_k = nn.Linear(embedding_dim, embedding_dim, bias=qkv_bias)
        self.W_v = nn.Linear(embedding_dim, embedding_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.last_attn_weights = None

    def forward(self, x):
        B, T, D = x.shape

        # (B, T, D) -> (B, T, H, d_k) -> (B, H, T, d_k): split D into H heads of size d_k
        q = self.W_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        out, attn_weights = scaled_dot_product_attention(q, k, v, causal=self.causal)  # (B, H, T, d_k)
        self.last_attn_weights = attn_weights

        # concatenate heads back into one D-dim vector per position
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.dropout(self.out_proj(out))
