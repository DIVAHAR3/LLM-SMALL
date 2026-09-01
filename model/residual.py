import torch.nn as nn


class ResidualConnection(nn.Module):
    """Wraps a sublayer so its output is ADDED to its input rather than
    replacing it: output = x + sublayer(x). This gives gradients a direct
    shortcut path back through every block during backpropagation, and lets
    a block learn a small adjustment to its input rather than having to
    reconstruct the whole representation from scratch."""

    def __init__(self, sublayer):
        super().__init__()
        self.sublayer = sublayer

    def forward(self, x):
        return x + self.sublayer(x)
