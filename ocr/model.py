"""Convolutional character classifier: a small CNN that takes one
image_size x image_size grayscale character crop and predicts which of
CHARACTERS it is.

Why a CNN, not the attention architecture this project's GPT already
uses: attention treats its input as a sequence of tokens with no assumed
spatial structure between them -- exactly right for text, wrong for an
image. A character crop's meaning comes from LOCAL pixel neighborhoods (a
stroke here, a curve there) that mean the same thing wherever they sit in
the crop. Convolution encodes that directly: a small learned filter
slides across every position, reusing the SAME weights everywhere
("parameter sharing") -- a stroke detector learned in one corner works
just as well in another, unlike a fully-connected layer, which would
have to learn that pattern separately at every pixel location.

Max-pooling then shrinks the spatial size, keeping only the strongest
response in each small region -- this is what gives CNNs tolerance to
the character being drawn a couple pixels off-center, exactly the
position jitter ocr/synthetic_data.py deliberately introduces.

Stacking two conv+pool stages builds up from local, low-level patterns
(edges, curves) to whole-character shapes, before a final fully-connected
head maps the resulting features to one of num_classes -- the same
cross-entropy classification objective already taught in Phase 11, just
one prediction per image instead of one per sequence position."""
import torch.nn as nn

_CONFIG_KEYS = {"image_size", "num_classes", "conv1_channels", "conv2_channels", "hidden_dim", "dropout"}
_REQUIRED_POSITIVE_INT_KEYS = ("image_size", "num_classes", "conv1_channels", "conv2_channels", "hidden_dim")


def validate_ocr_model_config(config):
    """Mirrors model/gpt.py's validate_model_config: catch a bad config
    with one clear message before any submodule is built."""
    missing = [k for k in _REQUIRED_POSITIVE_INT_KEYS if k not in config]
    if missing:
        raise ValueError(f"OCR model config missing required key(s): {missing}")

    for key in _REQUIRED_POSITIVE_INT_KEYS:
        value = config[key]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"OCR model config '{key}' must be a positive integer, got {value!r}")

    if config["image_size"] % 4 != 0:
        raise ValueError(
            f"OCR model config 'image_size' ({config['image_size']}) must be divisible by 4 "
            "-- two 2x2 max-pool stages halve it twice"
        )

    dropout = config.get("dropout", 0.0)
    if not isinstance(dropout, (int, float)) or isinstance(dropout, bool) or not (0.0 <= dropout < 1.0):
        raise ValueError(f"OCR model config 'dropout' must be a number in [0, 1), got {dropout!r}")


class CharacterCNN(nn.Module):
    def __init__(self, image_size, num_classes, conv1_channels=16, conv2_channels=32, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.config = {
            "image_size": image_size,
            "num_classes": num_classes,
            "conv1_channels": conv1_channels,
            "conv2_channels": conv2_channels,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
        }

        self.conv1 = nn.Conv2d(1, conv1_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        pooled_size = image_size // 4  # two 2x2 pool stages, each halving the spatial size
        self.fc1 = nn.Linear(conv2_channels * pooled_size * pooled_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    @classmethod
    def from_config(cls, config):
        """Build a model straight from an ocr_model_config.json-shaped
        dict, ignoring any non-constructor keys (e.g. "_comment").
        Validates the config first -- this is the intended entry point
        for every real (non-test) construction, same pattern as
        GPTModel.from_config (Phase 28)."""
        validate_ocr_model_config(config)
        kwargs = {k: v for k, v in config.items() if k in _CONFIG_KEYS}
        return cls(**kwargs)

    def forward(self, images):
        """images: (B, 1, image_size, image_size) -> logits (B, num_classes)"""
        x = self.pool(self.relu(self.conv1(images)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.flatten(1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def summary(self):
        config_str = ", ".join(f"{k}={v}" for k, v in self.config.items())
        lines = [f"CharacterCNN({config_str})"]
        for name, module in self.named_children():
            n = sum(p.numel() for p in module.parameters())
            lines.append(f"  {name:<10} {n:>10,} params")
        lines.append(f"  {'TOTAL':<10} {self.num_parameters():>10,} params")
        return "\n".join(lines)
