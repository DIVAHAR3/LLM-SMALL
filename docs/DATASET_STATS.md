# Phase 24 — Data Pipeline Hardening

`training/data_prep.py` gained a document-aware, hardened pipeline: raw text → paragraph-level "documents" → cleaned (line-ending + **Unicode NFC normalization**, new this phase) → malformed/too-short documents filtered → **exact-match deduplicated** → reproducible **seeded, document-level** train/val split → tokenized. `scripts/prepare_data.py` now accepts multiple `--raw-file` arguments and reports full dataset stats.

**The canonical pipeline default is unchanged** (`data/raw/placeholder_corpus.txt` alone, same `configs/model_config.json`/`tokenizer/vocab.json` outputs) — `checkpoints/phase13_run.pt` and the live API remain valid and untouched. The demonstration below deliberately used isolated output paths (`data/processed_hardening_demo/`, `tokenizer/vocab_hardening_demo.json`), the same pattern Phase 16's experiment used, specifically so it couldn't corrupt the live tokenizer↔checkpoint pairing.

## Why paragraph-level documents

The pipeline needed a unit smaller than "the whole file" to deduplicate and split. Paragraphs were a deliberate, non-arbitrary choice here: `data/raw/experiment1_larger_corpus.txt` (Phase 16) was authored by literally extending `data/raw/placeholder_corpus.txt`'s own paragraphs verbatim — so combining both files and deduplicating at paragraph granularity gives a **real** overlap to find, not a contrived test case.

## Real demonstration: combining both existing corpus files

```
.venv\Scripts\python.exe scripts\prepare_data.py \
  --raw-file data\raw\placeholder_corpus.txt data\raw\experiment1_larger_corpus.txt \
  --processed-dir data\processed_hardening_demo \
  --tokenizer-path tokenizer\vocab_hardening_demo.json \
  --model-config configs\hardening_demo_model_config.json  # scratch copy, deleted after
```

Result:

| Stat | Value |
|---|---|
| Source files | `placeholder_corpus.txt` + `experiment1_larger_corpus.txt` |
| Documents (paragraphs) after filtering | 20 |
| **Duplicates removed** | **7** — matches exactly the count of paragraphs known to be copied verbatim between the two files |
| Malformed/too-short removed | 0 (our prose paragraphs are all substantial) |
| Total chars (deduped documents) | 21,440 |
| Avg / min / max document length (chars) | 1,072 / 386 / 1,749 |
| Vocab size | 59 (identical to `experiment1_larger_corpus.txt` alone — expected, since it's a strict superset of `placeholder_corpus.txt`'s characters) |
| Train tokens | 18,403 |
| Val tokens | 3,073 |
| Total tokens | 21,476 |
| Seed | 1337 (reproducible — same seed always gives the same document-level train/val assignment) |

The duplicate count (7) landing exactly where expected is a genuine correctness proof, not just a plausible-looking number — this is real dedup working on real, known overlapping content.

## What changed in `clean_text`

Added Unicode NFC normalization. Rationale: a character-level tokenizer treats different Unicode representations of the same visible character (e.g., precomposed "é" vs. "e" + combining acute accent) as *different* vocabulary entries — without normalization, a corpus mixing forms would silently fragment its own vocabulary and statistics. Verified via a dedicated test using `unicodedata.normalize("NFD", ...)` to construct a genuinely decomposed input at runtime (embedding two different Unicode forms as literal source text turned out to be unreliable — some part of the file-writing pipeline silently collapsed them to one form in transit, an interesting finding in its own right, worked around by computing the decomposed form programmatically instead).

## Split granularity: reproducible document-level, not character-level

The original Phase 4 split (`split_ids`) was contiguous over a single character stream — deterministic, but not document-aware, so it can't coexist with deduplication (dedup operates on documents; a character-stream split has no document boundaries to respect). `split_documents` shuffles a *copy* of the document list with a `random.Random(seed)` instance and splits at the document level, so the same seed always reproduces the same train/val assignment, and no single document's content is ever divided across both splits. `split_ids` is kept (still tested, still valid) for simple single-stream use cases that don't need document-level granularity.
