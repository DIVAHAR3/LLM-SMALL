import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Normalizes each position's feature vector independently: for every
    token, across its embedding_dim features, subtract the mean and divide
    by the standard deviation, then apply a learned per-feature scale
    (gamma) and shift (beta). Nothing is computed across the batch or
    across other positions in the sequence -- each token is normalized
    purely against itself, which is what makes LayerNorm well suited to
    sequence models where batch size and sequence length can vary."""

    def __init__(self, embedding_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(embedding_dim))
        self.beta = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * normalized + self.beta
