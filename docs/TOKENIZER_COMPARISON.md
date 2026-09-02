# Phase 25 — Subword Tokenizer

`tokenizer/bpe_tokenizer.py`: `BPETokenizer`, byte-pair encoding implemented from scratch (matching this project's pattern of hand-rolling core concepts), same interface as `CharTokenizer` (`encode`/`decode`/`vocab_size`/`save`/`load`) for a direct comparison. Full write-up of BPE itself is in this session's teaching notes; summary: start from individual characters, repeatedly merge the most frequent adjacent pair into a new token, until reaching a target vocab size. Encoding replays the same merges in the order they were learned. No pre-tokenized word ever needed a true `<UNK>` fallback in testing — the base character vocabulary is always available as a worst case.

## Real comparison, `data/raw/experiment1_larger_corpus.txt` (21,479 characters)

| tokenizer | vocab_size | tokens | chars/token | train time |
|---|---:|---:|---:|---:|
| char-level (Phase 3) | 59 | 21,479 | 1.00 | — |
| BPE (v=150) | 150 | 14,064 | 1.53 | 0.29s |
| BPE (v=300) | 300 | 11,742 | 1.83 | 0.70s |
| BPE (v=500) | 500 | 10,268 | 2.09 | 1.23s |

At vocab_size=300, BPE needs **45% fewer tokens** than char-level for the same text — meaning our fixed 128-token context window would cover roughly **234 characters** of actual text instead of 128, without changing the model's architecture at all. BPE training itself is cheap at our corpus scale (under 1.3s even at vocab_size=500).

**Concrete illustration** — encoding `"the model"`:
```
char-level:   [52, 40, 37, 5, 45, 47, 36, 37, 44]              -- 9 tokens, one per character
BPE (v=300):  [65, 5, 101]  ->  "the", " ", "model"            -- 3 tokens, both words merged whole
```
At just 300 merges, BPE has already learned "the" and "model" as single tokens from a 21KB corpus — a direct, visible demonstration of what those merges are actually doing.

## Recommendation: adopt BPE (vocab_size ≈ 300) for future training — with a real tradeoff to weigh

The token-efficiency gain is substantial and directly addresses this project's most binding constraint so far (a 128-token context window on a tiny model, repeatedly shown across Phase 13/16/22 to be where the model runs out of room). The extra embedding parameters are negligible: `(300-59) × 128 × 2 ≈ 61,696` params, next to nothing against the 821K total.

**But switching isn't free**, and matches the same shape of decision as Phase 23's GPU finding: `checkpoints/phase13_run.pt` and `experiment1_more_data.pt` have embedding tables keyed to the exact character→id mapping of the current char-level tokenizer. Swapping tokenizers means those IDs mean something different (or nothing) under a new vocabulary — **it requires retraining from scratch**, not just swapping a file. That's a real, non-trivial follow-up decision, not something to fold silently into a measurement/comparison phase.

Per this phase's stop condition ("pick one with justification"): **BPE is the better choice going forward**, justified by the measured 45% token reduction on real project text. Whether to actually cut over the canonical pipeline now (retraining a fresh checkpoint) or continue with char-level a while longer is deliberately left as an explicit follow-up decision, not decided here.
