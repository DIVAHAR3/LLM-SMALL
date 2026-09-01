import torch
from torch.optim import AdamW

from training.checkpoint import load_checkpoint, save_checkpoint
from training.loss import cross_entropy_loss


def evaluate(model, val_loader, max_batches=None):
    was_training = model.training
    model.eval()
    total_loss, num_batches = 0.0, 0
    try:
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                if max_batches is not None and i >= max_batches:
                    break
                loss = cross_entropy_loss(model(x), y)
                total_loss += loss.item()
                num_batches += 1
    finally:
        model.train(was_training)
    return total_loss / max(num_batches, 1)


def train(
    model,
    train_loader,
    val_loader,
    config,
    max_steps=None,
    checkpoint_path=None,
    resume_from=None,
    eval_every=None,
    eval_max_batches=5,
    log_fn=print,
):
    """AdamW training loop with gradient accumulation, gradient clipping,
    periodic validation, and checkpointing/resume.

    config must contain: learning_rate, weight_decay, epochs,
    grad_accumulation_steps, grad_clip_norm (set to None to disable
    clipping).

    Epoch bookkeeping is coarse: resuming restarts the checkpointed epoch
    from its beginning rather than resuming mid-epoch. True mid-epoch
    resume would require also checkpointing the DataLoader/sampler state,
    which is unnecessary complexity at this dataset's scale.
    """
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    start_epoch, global_step = 0, 0
    history = {"train_loss": [], "val_loss": []}

    if resume_from is not None:
        checkpoint = load_checkpoint(resume_from, model, optimizer)
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["step"]
        history = checkpoint.get("metrics", history)
        log_fn(f"Resumed from {resume_from}: epoch={start_epoch}, step={global_step}")

    accumulation_steps = config.get("grad_accumulation_steps", 1)
    grad_clip_norm = config.get("grad_clip_norm")

    model.train()
    epoch = start_epoch
    stop = False
    for epoch in range(start_epoch, config["epochs"]):
        optimizer.zero_grad()
        for batch_idx, (x, y) in enumerate(train_loader):
            logits = model(x)
            loss = cross_entropy_loss(logits, y)
            # Divide before backward so accumulated micro-batch gradients
            # average to the same scale as one large batch, not sum to
            # accumulation_steps times too large.
            (loss / accumulation_steps).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                history["train_loss"].append(loss.item())

                if eval_every is not None and global_step % eval_every == 0:
                    val_loss = evaluate(model, val_loader, max_batches=eval_max_batches)
                    history["val_loss"].append(val_loss)
                    log_fn(f"step {global_step}: train_loss={loss.item():.4f} val_loss={val_loss:.4f}")

                if max_steps is not None and global_step >= max_steps:
                    stop = True
                    break
        if stop:
            break

    if checkpoint_path is not None:
        save_checkpoint(checkpoint_path, model, optimizer, epoch, global_step, config, history)
        log_fn(f"Saved checkpoint to {checkpoint_path}")

    return history
