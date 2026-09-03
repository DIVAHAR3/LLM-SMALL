"""Phase 26: instruction-tune the pretrained checkpoint on a small,
original instruction dataset. Run from the project root:

    .venv\\Scripts\\python.exe scripts\\finetune_instructions.py
"""
import json
import sys
import time
from pathlib import Path

from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tokenizer.char_tokenizer import CharTokenizer  # noqa: E402
from training.checkpoint import load_for_inference  # noqa: E402
from training.data_prep import clean_text, split_documents, split_into_paragraphs  # noqa: E402
from training.dataset import TextDataset  # noqa: E402
from training.reproducibility import set_seed  # noqa: E402
from training.train import train  # noqa: E402

BASE_CHECKPOINT = ROOT / "checkpoints" / "phase13_run.pt"
TOKENIZER_PATH = ROOT / "tokenizer" / "vocab.json"
INSTRUCTIONS_PATH = ROOT / "data" / "raw" / "instructions.txt"
OUTPUT_CHECKPOINT = ROOT / "checkpoints" / "phase26_instruction_tuned.pt"

MAX_STEPS = 300
FINE_TUNE_LR = 5e-5  # deliberately lower than pretraining's 3e-4, to adapt rather than overwrite
SEED = 1337

# Model weights here come from an already-trained checkpoint, not fresh
# random init, but the fine-tuning process itself (DataLoader shuffling,
# dropout) still draws from PyTorch's global RNG -- seed before any of
# that starts, same principle as scripts/train_ocr.py.
set_seed(SEED)

# Reuse the EXISTING tokenizer -- must not rebuild it, or the fine-tuned
# model's token ids would no longer match the pretrained embedding table.
tokenizer = CharTokenizer.load(str(TOKENIZER_PATH))

raw_text = INSTRUCTIONS_PATH.read_text(encoding="utf-8")
documents = [clean_text(p) for p in split_into_paragraphs(raw_text)]
train_docs, val_docs = split_documents(documents, val_split_ratio=0.15, seed=SEED)

train_ids = tokenizer.encode("\n\n".join(train_docs))
val_ids = tokenizer.encode("\n\n".join(val_docs))
print(f"instruction pairs: {len(documents)} ({len(train_docs)} train / {len(val_docs)} val)")
print(f"train_tokens={len(train_ids)}  val_tokens={len(val_ids)}")

model, base_checkpoint = load_for_inference(str(BASE_CHECKPOINT))
context_length = model.config["context_length"]
print(f"base checkpoint: {BASE_CHECKPOINT.name}  step={base_checkpoint['step']}  params={model.num_parameters():,}")

train_loader = DataLoader(TextDataset(train_ids, context_length), batch_size=32, shuffle=True)
val_loader = DataLoader(TextDataset(val_ids, context_length), batch_size=32, shuffle=False)

training_config = json.loads((ROOT / "configs" / "training_config.json").read_text(encoding="utf-8"))
fine_tune_config = {**training_config, "learning_rate": FINE_TUNE_LR}

print(f"\nfine-tuning: max_steps={MAX_STEPS}  lr={FINE_TUNE_LR}  batch_size=32  context_length={context_length}")
t0 = time.time()
history = train(
    model, train_loader, val_loader, fine_tune_config,
    max_steps=MAX_STEPS, eval_every=25, eval_max_batches=None,
    checkpoint_path=str(OUTPUT_CHECKPOINT),
)
elapsed = time.time() - t0
print(f"\nelapsed: {elapsed:.1f}s")
print(f"final train_loss: {history['train_loss'][-1]:.4f}")
print(f"final val_loss: {history['val_loss'][-1]:.4f}")
print(f"saved to {OUTPUT_CHECKPOINT}")
