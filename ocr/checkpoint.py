"""OCR-specific equivalent of training/checkpoint.py's
load_for_inference -- that function reconstructs the model architecture
via GPTModel.from_config specifically, so this project's second trained
model (CharacterCNN) needs its own inference-load path. The checkpoint
FORMAT is identical either way (training.checkpoint.save_checkpoint is
already model-agnostic -- ocr/train.py already reuses it directly), only
which model class to reconstruct differs."""
import torch

from ocr.model import CharacterCNN


def load_ocr_model_for_inference(path, map_location="cpu"):
    """Returns (model, checkpoint), model already in eval mode. Raises
    a clear error if the checkpoint has no embedded model_config."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if "model_config" not in checkpoint:
        raise ValueError(
            f"{path} has no embedded model_config -- cannot reconstruct the OCR model "
            "architecture automatically from this checkpoint alone."
        )
    model = CharacterCNN.from_config(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
