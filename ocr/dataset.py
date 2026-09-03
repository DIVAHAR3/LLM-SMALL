"""Converts the synthetic (PIL.Image, char) pairs from
ocr/synthetic_data.py into batched tensors CharacterCNN can train on."""
import torch
from torch.utils.data import Dataset

from ocr.synthetic_data import CHARACTERS

CHAR_TO_INDEX = {char: i for i, char in enumerate(CHARACTERS)}
INDEX_TO_CHAR = {i: char for char, i in CHAR_TO_INDEX.items()}


def image_to_tensor(image):
    """Grayscale PIL.Image -> (1, H, W) float tensor, pixel values
    scaled from [0, 255] to [0.0, 1.0]. Built from image.getdata() (a
    plain Python list of ints) rather than numpy -- this project has
    deliberately avoided adding numpy as a dependency (see
    analysis/image_analysis.py), and these crops are tiny (28x28 =
    784 pixels), so a pure-Python conversion is fast enough and keeps
    the dependency list minimal."""
    width, height = image.size
    pixels = list(image.convert("L").get_flattened_data())
    tensor = torch.tensor(pixels, dtype=torch.float32).reshape(1, height, width)
    return tensor / 255.0


class CharacterDataset(Dataset):
    """Wraps a list of (PIL.Image, char) pairs -- e.g. straight from
    ocr.synthetic_data.generate_dataset() -- as a torch Dataset."""

    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        image, char = self.examples[index]
        return image_to_tensor(image), CHAR_TO_INDEX[char]
