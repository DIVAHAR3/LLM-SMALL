import torch


def save_checkpoint(path, model, optimizer, epoch, step, config, metrics):
    """A checkpoint is self-describing: model weights, optimizer state
    (Adam's running averages -- resuming without this would reset momentum
    and shock training), epoch/step counters, the config that produced this
    model, and metrics logged so far."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "config": config,
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    """weights_only=False is safe here: checkpoints are always files this
    project wrote itself locally, never downloaded or otherwise untrusted."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
