# Phase 27 — Chat Format

## The problem: role tokens don't fit in a character vocabulary

Every phase so far has fed the model either raw text or `instruction:`/`response:` as plain literal characters. A real chat format needs distinct, unambiguous markers for who is speaking — `<SYSTEM>`, `<USER>`, `<ASSISTANT>` — that the model can learn to treat as structural boundaries, not just more text.

The existing `CharTokenizer`'s vocabulary (`tokenizer/vocab.json`, 51 tokens) has no `<` or `>` characters at all. Writing `<SYSTEM>` as literal text would encode it as a run of individual, mostly out-of-vocabulary characters — not one recognizable unit. The fix is to make each role token a genuine **atomic special token**, the same way `<BOS>`/`<EOS>`/`<PAD>`/`<UNK>` already are: one id, inserted directly into the id sequence, never produced by character-by-character encoding.

## Extending a pretrained vocabulary safely

Two things have to grow together, and both must preserve everything already learned:

**Tokenizer side** — `CharTokenizer.with_additional_special_tokens()` returns a *new* tokenizer with the new tokens **appended at the end** of the vocabulary. Every existing character keeps its exact original id. Inserting in the middle would shift every later id and silently corrupt the correspondence between embedding rows and the characters they were actually trained on — appending is the only safe option.

**Model side** — `GPTModel.resize_vocab()` grows the token-embedding table and the LM head to the new vocab size, in place. The first `old_vocab_size` rows are copied byte-for-byte from the pretrained weights; only the new rows (one per new token) get fresh random initialization. Old ids still point at the same, unchanged embedding rows they were trained on — the model doesn't forget anything, it just gains a few new, untrained rows for the new tokens.

Both are covered by tests that check the mechanism directly rather than trusting behavior: existing ids/weights unchanged after extension, new rows present but not just zeros, forward/backward passes work with the new ids, shrinking is rejected, growing to the same size is a no-op.

## Building chat examples as token ids, not text

`training/chat_format.py` builds each training example directly as an id list:

```
<SYSTEM> system_msg <USER> user_msg <ASSISTANT> assistant_msg
```

`build_prompt_ids()` produces the same thing truncated right after `<ASSISTANT>` — that's the actual inference prompt; generation fills in everything after it.

## Fine-tuning run

Starting point: Phase 26's instruction-tuned checkpoint (`phase26_instruction_tuned.pt`). Same recipe as Phase 26 — 300 steps, batch_size=32, context_length=128, lr=5e-5 — the only new step is extending the tokenizer (51 → 54 tokens) and resizing the model (821,248 → 822,016 params; +768 = 3 new tokens × 128 embedding_dim × 2 matrices, confirming exactly the right rows were added) before training starts.

Dataset: `data/raw/chat_examples.jsonl`, 21 `(system, user, assistant)` triples reformatted from Phase 26's instruction pairs — same content, now in chat form. Split via Phase 24's document-level splitter (each example is one document), seed=1337: 17 train / 4 val.

```
step  25: train_loss=2.3967  val_loss=2.4820
step 300: train_loss=2.1571  val_loss=2.2910
elapsed: 191.8s
```

Loss dropped and train/val moved together — no divergence, consistent with every prior fine-tuning run in this project.

## A real bug, found by testing past the easy case

Initial `inference.chat.chat()` only treated `<SYSTEM>` and `<USER>` as stop tokens — the reasoning being "stop when the model tries to start a new turn." At `max_new_tokens=100`, four test prompts against the real trained checkpoint looked fine.

Raising `max_new_tokens` to 400 on the same prompts exposed the gap: 3 of 4 replies contained literal, visible `"<ASSISTANT>"` text mid-reply. The model — unsurprisingly, given 821K parameters and ~3,600 training tokens — sometimes hallucinates a second, malformed `<ASSISTANT>` marker instead of naturally trailing off. Since that id wasn't in the stop set, generation ran past it, and `decode(skip_special_tokens=True)` only strips the original four special tokens, not dynamically-added role tokens — so it printed as plain text.

**Fix**: all three role tokens are stop tokens, not just two. A hallucinated second `<ASSISTANT>` is just as much "past the end of this reply" as a new user turn would be:

```python
stop_ids = {tokenizer.char_to_id[SYSTEM_TOKEN], tokenizer.char_to_id[USER_TOKEN], tokenizer.char_to_id[ASSISTANT_TOKEN]}
```

The pre-existing test suite had a test asserting `"<ASSISTANT>" not in reply`, and it passed — using a randomly-initialized model whose particular random seed just never happened to trigger the bug. That's a coincidental pass, not a verified mechanism. Added a deterministic regression test using a test double forced to always predict the `<ASSISTANT>` id, which fails without the fix and passes with it regardless of seed.

Re-ran the same 400-token, 4-prompt check against the real checkpoint after the fix: no reply contains `<ASSISTANT>`, `<SYSTEM>`, or `<USER>` text. Output is still character-soup — expected at this scale, same honest caveat as Phase 26 — but the structural leak is gone.

## What this shows

The chat-format *mechanism* — atomic role tokens, safe vocabulary/embedding extension, id-list example construction, and halting generation on any role-token reappearance — works correctly and is now verified by both a fast deterministic unit test and a real end-to-end run against the trained model. As with instruction tuning, genuine coherent conversation is a scale question this project's model size was never going to answer; what's verified here is that the *format and stopping logic* are correct, including a real bug that only a long-generation stress test surfaced.
