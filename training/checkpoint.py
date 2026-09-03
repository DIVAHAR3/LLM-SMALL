import torch

from model.gpt import GPTModel
from training.reproducibility import capture_run_metadata


def save_checkpoint(path, model, optimizer, epoch, step, config, metrics):
    """A checkpoint is self-describing: model weights AND the model
    architecture config (model.config, set by GPTModel at construction --
    vocab_size, context_length, embedding_dim, etc.), optimizer state
    (Adam's running averages -- resuming without this would reset momentum
    and shock training), epoch/step counters, the training config that
    produced this run, and metrics logged so far. Storing model_config
    alongside model_state_dict is what makes load_for_inference() able to
    reconstruct the correct model from the checkpoint file alone, with no
    separate config file required.

    Also self-describing about HOW this run could be reproduced (Phase
    29): git commit, dataset provenance, and software/hardware
    environment, captured automatically for every checkpoint -- callers
    don't need to remember to ask for it. The seed itself is only
    meaningful if the caller actually applied it via
    training.reproducibility.set_seed() before constructing the model;
    recording it here doesn't retroactively make an unseeded run
    reproducible."""
    torch.save(
        {
            "model_config": model.config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "training_config": config,
            "metrics": metrics,
            "reproducibility": capture_run_metadata(seed=config.get("seed")),
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    """Loads into an ALREADY-CONSTRUCTED model/optimizer -- the resume path,
    where the caller already knows the architecture. weights_only=False is
    safe here: checkpoints are always files this project wrote itself
    locally, never downloaded or otherwise untrusted."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def load_for_inference(path, map_location="cpu"):
    """The inference-load path: reconstructs the model architecture from the
    checkpoint's own embedded model_config, with no separate config file
    needed at all. Returns (model, checkpoint), model already in eval mode.

    Raises a clear error for checkpoints saved before model_config was
    stored (i.e. before this function existed) -- there is no way to
    recover the architecture from an older checkpoint automatically; it
    would need to be reconstructed from the training run that produced it.
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if "model_config" not in checkpoint:
        raise ValueError(
            f"{path} has no embedded model_config (saved before Phase 17). "
            "Cannot reconstruct the model architecture automatically from this "
            "checkpoint alone -- rebuild it via the training run/config that "
            "produced it, or use load_checkpoint(path, model) with a manually "
            "constructed model instead."
        )
    model = GPTModel.from_config(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
