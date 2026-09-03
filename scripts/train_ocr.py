"""OCR sub-phase 2b: generate the synthetic character dataset, train
CharacterCNN on it, save a checkpoint. Run from the project root:

    .venv\\Scripts\\python.exe scripts\\train_ocr.py
"""
import json
import sys
import time
from pathlib import Path

from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ocr.dataset import CharacterDataset  # noqa: E402
from ocr.model import CharacterCNN  # noqa: E402
from ocr.synthetic_data import generate_dataset  # noqa: E402
from ocr.train import train  # noqa: E402
from training.data_prep import split_documents  # noqa: E402

MODEL_CONFIG_PATH = ROOT / "configs" / "ocr_model_config.json"
OUTPUT_CHECKPOINT = ROOT / "checkpoints" / "ocr_character_cnn.pt"

SAMPLES_PER_CHARACTER = 40
VAL_SPLIT_RATIO = 0.15
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 1e-3
SEED = 1337


def main():
    model_config = json.loads(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
    model = CharacterCNN.from_config(model_config)
    print(f"model: {model.num_parameters():,} params")

    examples = generate_dataset(samples_per_character=SAMPLES_PER_CHARACTER, seed=SEED)
    # split_documents is generic (shuffle-then-split any list) -- reused
    # as-is from the text pipeline (Phase 24), nothing OCR-specific about it
    train_examples, val_examples = split_documents(examples, val_split_ratio=VAL_SPLIT_RATIO, seed=SEED)
    print(f"dataset: {len(examples)} examples ({len(train_examples)} train / {len(val_examples)} val)")

    train_loader = DataLoader(CharacterDataset(train_examples), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(CharacterDataset(val_examples), batch_size=BATCH_SIZE, shuffle=False)

    print(f"\ntraining: epochs={EPOCHS} lr={LEARNING_RATE} batch_size={BATCH_SIZE}")
    t0 = time.time()
    history = train(
        model, train_loader, val_loader,
        learning_rate=LEARNING_RATE, epochs=EPOCHS, checkpoint_path=str(OUTPUT_CHECKPOINT),
    )
    elapsed = time.time() - t0
    print(f"\nelapsed: {elapsed:.1f}s")
    print(f"final train_loss: {history['train_loss'][-1]:.4f}")
    print(f"final val_loss: {history['val_loss'][-1]:.4f}")
    print(f"final val_accuracy: {history['val_accuracy'][-1]:.2%}")


if __name__ == "__main__":
    main()
