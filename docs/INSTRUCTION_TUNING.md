# Phase 26 — Instruction Tuning

## Pretraining vs. instruction tuning

Pretraining (Phase 12-16) trains on raw, unstructured text with one objective: predict the next token. Instruction tuning takes an **already-pretrained** model and continues training — same objective (still next-token cross-entropy), same architecture, same optimizer type — on a smaller, specially-formatted dataset of `(instruction, response)` pairs. Nothing about the training mechanism changes; what changes is the data format and the fact that it's a second, targeted phase built on existing weights, not training from scratch.

**Honesty check going in**: at 821K parameters and a ~3.8KB instruction dataset, genuine "instruction following" (different questions producing different, relevant answers) was never a realistic goal. The real question was narrower: does the fine-tuning mechanism visibly shift the model's behavior at all, in a direction attributable to the new data?

## The dataset

`data/raw/instructions.txt` — 21 original `(instruction, response)` pairs about this project's own concepts (tokenizers, attention, checkpoints, sampling, etc.), 3,802 characters. Written entirely in lowercase, deliberately: the existing tokenizer's vocabulary (built from Phase 4's original corpus) only contains a handful of uppercase letters and no digits or apostrophes — verified character-by-character before writing (`out-of-vocab chars: NONE`) rather than discovering `<UNK>` tokens after the fact.

## Fine-tuning setup

Same architecture, same 300 steps, same batch_size=32/context_length=128 as Phase 13's baseline run — the only real variables are the data and a **lower learning rate (5e-5 vs. pretraining's 3e-4)**, deliberately smaller to adapt the model rather than overwrite its pretrained weights. Started from `checkpoints/phase13_run.pt`'s weights only (fresh optimizer state — a new, very different, much smaller data distribution doesn't obviously benefit from carrying over Adam's momentum tuned on the original corpus). Used Phase 24's document-level split (each instruction/response pair as one document) rather than a contiguous character split, so no pair straddles the train/val boundary.

Loss dropped cleanly and stayed stable — train 2.4609 → 2.2445, val 2.5001 → 2.3437, moving together the whole run, no divergence:

```
step  25: train=2.4609 val=2.5001
step 150: train=2.2570 val=2.3694
step 300: train=2.2445 val=2.3437
```

Saved as `checkpoints/phase26_instruction_tuned.pt`.

## Before/after comparison

Same prompts, same seed, both models, temperature=0.7:

| Prompt | Before (phase13_run.pt) | After (phase26_instruction_tuned.pt) |
|---|---|---|
| `instruction: what is a checkpoint?\nresponse:` (seen in training) | `text tud thal...heng s` | `...bion: des **chesponstistaton**` |
| `instruction: what is gradient clipping?\nresponse:` (held-out val) | `text tud thal...heng s` | `...bin: odes **chesponstistatos**` |
| `instruction: what is a gpu?\nresponse:` (never in the dataset) | `the atude toll...heng os` | `...bion: des **ch** tingig heng s` |
| `the model` (plain, non-instruction prompt) | `the ated thal...heng os` | `...bion: djy **chesponig** heng s` |

## What this actually shows

**A real, measurable, honest finding, not overclaimed:** three of the four AFTER outputs — including the plain `"the model"` prompt, which has nothing to do with instructions — contain a recognizable corrupted fragment of **"response:"** (`chespons`/`chespon`), one of only two fixed phrases repeated in every one of the 21 training pairs. None of the BEFORE outputs show this pattern at all. That's the fine-tuning mechanism visibly working: 300 steps at a low learning rate was enough for the model to pick up a strong statistical association with the new data's dominant recurring vocabulary, strong enough to bleed into generations regardless of prompt.

**What it does not show**: genuine instruction-following. The three instruction-formatted prompts (checkpoint / gradient clipping / gpu) don't produce meaningfully *different* answers from each other in either model — before or after, output is dominated by whichever corpus's own statistical habits are currently loaded, not by the specific question asked. That's exactly what was predicted going in: at this scale, the model can absorb a surface-level fingerprint of a new data distribution, but 821K parameters and ~3,200 training tokens isn't enough to learn per-question semantic mapping.

**Conclusion**: the instruction-tuning *mechanism* — continuing training on a reformatted, targeted dataset at a reduced learning rate — works exactly as designed and produces a measurable, attributable shift. Real instruction-following capability is a scale question, not a mechanism question, consistent with everything this project has found about its own model size at every phase so far.
