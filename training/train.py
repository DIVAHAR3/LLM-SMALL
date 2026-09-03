import json
import time
from pathlib import Path

import torch
from torch.optim import AdamW

from training.checkpoint import load_checkpoint, save_checkpoint
from training.loss import cross_entropy_loss
from training.monitoring import get_available_memory_mb


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
    log_path=None,
):
    """AdamW training loop with size-weighted gradient accumulation,
    gradient clipping, periodic validation, checkpointing/resume, and
    monitoring (Phase 30).

    config must contain: learning_rate, weight_decay, epochs,
    grad_accumulation_steps, grad_clip_norm (set to None to disable
    clipping).

    Monitoring: at each eval_every step (the same cadence validation
    already runs at -- there's no reason for these to be on different
    schedules), logs train/val loss, the optimizer's actual current
    learning rate, cumulative tokens processed, wall-clock elapsed time,
    and available system memory. If log_path is given, the same fields
    are ALSO appended as one JSON line per logged step -- lightweight,
    structured, plottable later (pandas/matplotlib) without adding a
    TensorBoard dependency this project doesn't otherwise need. Appends
    to an existing log_path rather than overwriting it, so a resumed
    run's history isn't lost -- delete/rotate the file yourself before a
    genuinely fresh run if you don't want old and new runs mixed in it.

    Gradient accumulation weights each micro-batch by its actual token
    count rather than a fixed 1/accumulation_steps split, so a window
    containing a ragged final batch still reproduces the same effective
    gradient as one large pooled batch would. Any leftover gradient from
    an incomplete window at the end of an epoch is flushed as one
    (smaller) optimizer step rather than silently discarded by the next
    epoch's zero_grad().

    Epoch bookkeeping: the saved epoch is the NEXT epoch to run --
    incremented past the current one only if it finished naturally (not
    cut short by max_steps), so resuming never silently re-trains a
    completed epoch's data. A max_steps-interrupted epoch does still
    restart from its own beginning on resume, since its exact DataLoader
    position isn't checkpointed -- that coarseness is an accepted
    tradeoff at this dataset's scale.

    Known limitation: no RNG state (dropout masks, DataLoader shuffle
    order) is checkpointed, so a resumed run's exact trajectory will
    diverge from what an uninterrupted run would have produced whenever
    dropout > 0 or shuffling is used, even though model/optimizer state
    match exactly at the resume point. Full reproducibility (seeding, RNG
    capture) is Phase 29's job, not this one.
    """
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    start_epoch, global_step = 0, 0
    cumulative_tokens = 0
    start_time = time.time()
    history = {
        "train_loss": [], "val_loss": [],
        "learning_rate": [], "tokens_processed": [], "elapsed_seconds": [], "memory_mb": [],
    }
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    if resume_from is not None:
        checkpoint = load_checkpoint(resume_from, model, optimizer)
        # optimizer.load_state_dict() overwrites every param_group field
        # (lr, weight_decay, ...) with the checkpointed values -- reapply
        # this call's config explicitly so a deliberately changed LR/weight
        # decay (e.g. a schedule) isn't silently discarded by resume.
        for group in optimizer.param_groups:
            group["lr"] = config["learning_rate"]
            group["weight_decay"] = config["weight_decay"]
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["step"]
        history = checkpoint.get("metrics", history)
        log_fn(f"Resumed from {resume_from}: epoch={start_epoch}, step={global_step}")

    accumulation_steps = config.get("grad_accumulation_steps", 1)
    grad_clip_norm = config.get("grad_clip_norm")

    def apply_step(window_summed_loss, window_token_count):
        for p in model.parameters():
            if p.grad is not None:
                p.grad /= window_token_count
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad()
        return window_summed_loss / window_token_count

    model.train()
    epoch = start_epoch
    epoch_fully_completed = False
    stop = False

    for epoch in range(start_epoch, config["epochs"]):
        optimizer.zero_grad()
        window_summed_loss, window_token_count, window_batches = 0.0, 0, 0
        epoch_fully_completed = False

        for x, y in train_loader:
            logits = model(x)
            loss = cross_entropy_loss(logits, y)
            token_count = y.numel()
            # Backward on the unnormalized per-window contribution; the
            # window's total gradient gets normalized by its true total
            # token count in apply_step, not by a fixed micro-batch count.
            (loss * token_count).backward()
            window_summed_loss += loss.item() * token_count
            window_token_count += token_count
            window_batches += 1
            cumulative_tokens += token_count

            if window_batches == accumulation_steps:
                avg_loss = apply_step(window_summed_loss, window_token_count)
                global_step += 1
                history["train_loss"].append(avg_loss)
                window_summed_loss, window_token_count, window_batches = 0.0, 0, 0

                if eval_every is not None and global_step % eval_every == 0:
                    val_loss = evaluate(model, val_loader, max_batches=eval_max_batches)
                    current_lr = optimizer.param_groups[0]["lr"]
                    elapsed = time.time() - start_time
                    memory_mb = get_available_memory_mb()

                    history["val_loss"].append(val_loss)
                    history["learning_rate"].append(current_lr)
                    history["tokens_processed"].append(cumulative_tokens)
                    history["elapsed_seconds"].append(elapsed)
                    history["memory_mb"].append(memory_mb)

                    mem_str = f"{memory_mb:.0f}MB" if memory_mb is not None else "unavailable"
                    log_fn(
                        f"step {global_step}: train_loss={avg_loss:.4f} val_loss={val_loss:.4f} "
                        f"lr={current_lr:.2e} tokens={cumulative_tokens:,} elapsed={elapsed:.1f}s mem_avail={mem_str}"
                    )

                    if log_path is not None:
                        record = {
                            "step": global_step, "train_loss": avg_loss, "val_loss": val_loss,
                            "learning_rate": current_lr, "tokens_processed": cumulative_tokens,
                            "elapsed_seconds": elapsed, "memory_mb": memory_mb,
                        }
                        with open(log_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(record) + "\n")

                if max_steps is not None and global_step >= max_steps:
                    stop = True
                    break
        else:
            epoch_fully_completed = True  # inner loop exhausted without break

        if window_batches > 0:
            avg_loss = apply_step(window_summed_loss, window_token_count)
            global_step += 1
            history["train_loss"].append(avg_loss)

        if stop:
            break

    if checkpoint_path is not None:
        saved_epoch = epoch + 1 if epoch_fully_completed else epoch
        save_checkpoint(checkpoint_path, model, optimizer, saved_epoch, global_step, config, history)
        log_fn(f"Saved checkpoint to {checkpoint_path}")

    return history
