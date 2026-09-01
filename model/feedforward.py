import torch.nn as nn

_ACTIVATIONS = {"gelu": nn.GELU, "relu": nn.ReLU}


class FeedForward(nn.Module):
    """Position-wise feed-forward network: the same Linear-Activation-Linear
    is applied independently to every position, with no mixing across
    positions (that's attention's job, not this module's)."""

    def __init__(self, embedding_dim, hidden_dim, dropout=0.0, activation="gelu"):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"unsupported activation '{activation}', choose from {list(_ACTIVATIONS)}")
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.activation = _ACTIVATIONS[activation]()
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return self.dropout(x)
