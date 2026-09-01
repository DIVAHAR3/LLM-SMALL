import torch
from torch.utils.data import DataLoader, Dataset


class TextDataset(Dataset):
    """Wraps a flat token-id sequence into fixed-length (input, target) windows
    for next-token prediction: target is the input window shifted one position right."""

    def __init__(self, ids, context_length):
        if len(ids) <= context_length:
            raise ValueError(
                f"Need more than context_length+1 tokens ({context_length + 1}), got {len(ids)}"
            )
        self.ids = ids
        self.context_length = context_length

    def __len__(self):
        return len(self.ids) - self.context_length

    def __getitem__(self, idx):
        x = torch.tensor(self.ids[idx : idx + self.context_length], dtype=torch.long)
        y = torch.tensor(self.ids[idx + 1 : idx + self.context_length + 1], dtype=torch.long)
        return x, y


def make_dataloaders(train_ids, val_ids, context_length, batch_size):
    train_ds = TextDataset(train_ids, context_length)
    val_ds = TextDataset(val_ids, context_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
