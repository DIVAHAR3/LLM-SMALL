import torch


def cross_entropy_loss(logits, targets):
    """Cross-entropy for next-token prediction.

    logits: (B, T, vocab_size) raw model output
    targets: (B, T) correct next-token ids

    Each position is an independent classification over the vocabulary
    (same "every position is its own problem" idea as the position-wise
    FFN). Loss at a position = -log(softmax(logits)[correct_class]);
    the final value is that averaged over every position in the batch.
    """
    B, T, V = logits.shape
    logits = logits.reshape(B * T, V)
    targets = targets.reshape(B * T)

    log_probs = torch.log_softmax(logits, dim=-1)                             # (B*T, V)
    correct_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B*T,)
    return -correct_log_probs.mean()
