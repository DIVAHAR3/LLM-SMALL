"""Training loop for CharacterCNN -- deliberately simpler than
training/train.py (the GPT's loop): this is single-label classification
over a small, fully in-memory synthetic dataset, so there's no need for
the GPT loop's size-weighted gradient accumulation (built for
variable-length token sequences) or its resume/checkpoint-mid-epoch
machinery. The classification loss itself is the same cross-entropy
concept already taught in Phase 11 (training/loss.py) -- one prediction
per image instead of one per sequence position -- so it's reused
directly via torch.nn.functional.cross_entropy rather than re-deriving
it. training.checkpoint.save_checkpoint IS reused directly, though: it's
already model-agnostic (just calls model.state_dict()/model.config), so
OCR checkpoints get the same self-describing format as everything else
in this project."""
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from training.checkpoint import save_checkpoint


def evaluate(model, val_loader):
    was_training = model.training
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    try:
        with torch.no_grad():
            for images, labels in val_loader:
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
                total_loss += loss.item() * labels.size(0)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)
    finally:
        model.train(was_training)
    return {"loss": total_loss / max(total, 1), "accuracy": correct / max(total, 1)}


def train(model, train_loader, val_loader, learning_rate=1e-3, epochs=10, checkpoint_path=None, log_fn=print):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    model.train()
    for epoch in range(epochs):
        epoch_loss, epoch_examples = 0.0, 0
        for images, labels in train_loader:
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * labels.size(0)
            epoch_examples += labels.size(0)

        train_loss = epoch_loss / max(epoch_examples, 1)
        val_metrics = evaluate(model, val_loader)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        log_fn(
            f"epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_accuracy={val_metrics['accuracy']:.2%}"
        )

    if checkpoint_path is not None:
        config = {"learning_rate": learning_rate, "epochs": epochs}
        save_checkpoint(checkpoint_path, model, optimizer, epoch=epochs, step=epochs, config=config, metrics=history)
        log_fn(f"Saved checkpoint to {checkpoint_path}")

    return history
