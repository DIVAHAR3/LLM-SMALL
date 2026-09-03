"""Phase 27: extend the tokenizer with <SYSTEM>/<USER>/<ASSISTANT> role
tokens, resize the model's embeddings to match, and fine-tune on a small
chat-formatted dataset, starting from Phase 26's instruction-tuned
checkpoint. Run from the project root:

    .venv\\Scripts\\python.exe scripts\\finetune_chat_format.py
"""
import json
import sys
import time
from pathlib import Path

from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tokenizer.char_tokenizer import CharTokenizer  # noqa: E402
from training.chat_format import ROLE_TOKENS, build_example_ids  # noqa: E402
from training.checkpoint import load_for_inference  # noqa: E402
from training.data_prep import split_documents  # noqa: E402
from training.dataset import TextDataset  # noqa: E402
from training.train import train  # noqa: E402

BASE_CHECKPOINT = ROOT / "checkpoints" / "phase26_instruction_tuned.pt"
BASE_TOKENIZER_PATH = ROOT / "tokenizer" / "vocab.json"
CHAT_TOKENIZER_PATH = ROOT / "tokenizer" / "vocab_chat.json"
CHAT_EXAMPLES_PATH = ROOT / "data" / "raw" / "chat_examples.jsonl"
OUTPUT_CHECKPOINT = ROOT / "checkpoints" / "phase27_chat_tuned.pt"

MAX_STEPS = 300
FINE_TUNE_LR = 5e-5

# Step 1: extend the tokenizer -- new role tokens appended at the end, every
# existing character id unchanged, so the pretrained embedding rows stay valid.
base_tokenizer = CharTokenizer.load(str(BASE_TOKENIZER_PATH))
tokenizer = base_tokenizer.with_additional_special_tokens(ROLE_TOKENS)
tokenizer.save(str(CHAT_TOKENIZER_PATH))
print(f"tokenizer: {base_tokenizer.vocab_size} -> {tokenizer.vocab_size} (+{len(ROLE_TOKENS)} role tokens)")

# Step 2: load the instruction-tuned model and resize its embeddings to match.
model, base_checkpoint = load_for_inference(str(BASE_CHECKPOINT))
print(f"base checkpoint: {BASE_CHECKPOINT.name}  step={base_checkpoint['step']}  vocab_size={model.config['vocab_size']}")
model.resize_vocab(tokenizer.vocab_size)
context_length = model.config["context_length"]
print(f"model resized: vocab_size={model.config['vocab_size']}  params={model.num_parameters():,}")

# Step 3: build chat-formatted training examples directly as token ids --
# NOT as text, since CharTokenizer encodes character-by-character and would
# split a literal "<SYSTEM>" string into out-of-vocabulary characters.
examples = [json.loads(line) for line in CHAT_EXAMPLES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
example_id_lists = [build_example_ids(tokenizer, ex["system"], ex["user"], ex["assistant"]) for ex in examples]

train_examples, val_examples = split_documents(example_id_lists, val_split_ratio=0.15, seed=1337)
train_ids = [tok_id for example in train_examples for tok_id in example]
val_ids = [tok_id for example in val_examples for tok_id in example]
print(f"chat examples: {len(examples)} ({len(train_examples)} train / {len(val_examples)} val)")
print(f"train_tokens={len(train_ids)}  val_tokens={len(val_ids)}")

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
