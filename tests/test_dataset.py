import sys
import unicodedata
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.data_prep import (
    build_tokenizer,
    clean_text,
    compute_dataset_stats,
    deduplicate_documents,
    filter_malformed_documents,
    prepare_corpus,
    split_documents,
    split_ids,
    split_into_paragraphs,
)
from training.dataset import TextDataset, make_dataloaders

ROOT = Path(__file__).resolve().parent.parent


class TestCleanText(unittest.TestCase):
    def test_normalizes_crlf(self):
        self.assertEqual(clean_text("a\r\nb\rc"), "a\nb\nc")

    def test_collapses_excess_blank_lines(self):
        self.assertEqual(clean_text("a\n\n\n\n\nb"), "a\n\nb")

    def test_strips_leading_trailing_whitespace(self):
        self.assertEqual(clean_text("  \n hello \n  "), "hello")

    def test_normalizes_unicode_to_nfc(self):
        # Computed at runtime via unicodedata, from a single reference
        # character (U+00E9, "e with acute") -- avoids embedding two
        # different Unicode forms as literal source text, which some part
        # of the file-writing pipeline was silently collapsing to one form.
        composed = "é"
        decomposed = unicodedata.normalize("NFD", composed)
        self.assertEqual(len(composed), 1)
        self.assertEqual(len(decomposed), 2)  # "e" + combining acute accent
        self.assertEqual(clean_text(decomposed), composed)
        self.assertEqual(len(clean_text(decomposed)), 1)


class TestSplitIds(unittest.TestCase):
    def test_split_is_contiguous_and_covers_all_ids(self):
        ids = list(range(100))
        train, val = split_ids(ids, val_split_ratio=0.1)
        self.assertEqual(train, ids[:90])
        self.assertEqual(val, ids[90:])
        self.assertEqual(len(train) + len(val), len(ids))

    def test_rejects_invalid_ratio(self):
        with self.assertRaises(ValueError):
            split_ids([1, 2, 3], val_split_ratio=1.5)


class TestSplitIntoParagraphs(unittest.TestCase):
    def test_splits_on_blank_lines(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        self.assertEqual(split_into_paragraphs(text), ["First paragraph.", "Second paragraph.", "Third paragraph."])

    def test_strips_each_paragraph_and_drops_whitespace_only_ones(self):
        text = "  para one  \n\n   \n\npara two"
        self.assertEqual(split_into_paragraphs(text), ["para one", "para two"])

    def test_normalizes_crlf_before_splitting(self):
        text = "para one\r\n\r\npara two"
        self.assertEqual(split_into_paragraphs(text), ["para one", "para two"])

    def test_tolerates_extra_blank_lines_between_paragraphs(self):
        text = "para one\n\n\n\n\npara two"
        self.assertEqual(split_into_paragraphs(text), ["para one", "para two"])


class TestDeduplicateDocuments(unittest.TestCase):
    def test_removes_exact_duplicates_preserving_first_occurrence_order(self):
        docs = ["a", "b", "a", "c", "b"]
        deduped, removed = deduplicate_documents(docs)
        self.assertEqual(deduped, ["a", "b", "c"])
        self.assertEqual(removed, 2)

    def test_no_duplicates_leaves_list_unchanged(self):
        docs = ["a", "b", "c"]
        deduped, removed = deduplicate_documents(docs)
        self.assertEqual(deduped, docs)
        self.assertEqual(removed, 0)


class TestFilterMalformedDocuments(unittest.TestCase):
    def test_drops_documents_shorter_than_min_chars(self):
        docs = ["short", "this one is long enough to keep"]
        kept, removed = filter_malformed_documents(docs, min_chars=10)
        self.assertEqual(kept, ["this one is long enough to keep"])
        self.assertEqual(removed, 1)

    def test_keeps_documents_at_or_above_threshold(self):
        docs = ["exactly10c"]
        kept, removed = filter_malformed_documents(docs, min_chars=10)
        self.assertEqual(kept, docs)
        self.assertEqual(removed, 0)


class TestSplitDocuments(unittest.TestCase):
    def test_same_seed_gives_identical_split(self):
        docs = [f"document {i} with enough content" for i in range(20)]
        train1, val1 = split_documents(docs, val_split_ratio=0.2, seed=42)
        train2, val2 = split_documents(docs, val_split_ratio=0.2, seed=42)
        self.assertEqual(train1, train2)
        self.assertEqual(val1, val2)

    def test_every_document_appears_exactly_once_across_both_splits(self):
        docs = [f"document {i}" for i in range(20)]
        train, val = split_documents(docs, val_split_ratio=0.25, seed=1)
        self.assertEqual(sorted(train + val), sorted(docs))
        self.assertEqual(len(set(train) & set(val)), 0)

    def test_rejects_invalid_ratio(self):
        with self.assertRaises(ValueError):
            split_documents(["a", "b"], val_split_ratio=0)


class TestComputeDatasetStats(unittest.TestCase):
    def test_returns_expected_values(self):
        docs = ["hello world", "a longer document here"]
        tokenizer = build_tokenizer(" ".join(docs))
        train_ids = tokenizer.encode(docs[0])
        val_ids = tokenizer.encode(docs[1])
        stats = compute_dataset_stats(docs, tokenizer, train_ids, val_ids, duplicates_removed=3, malformed_removed=1)
        self.assertEqual(stats["num_documents"], 2)
        self.assertEqual(stats["duplicates_removed"], 3)
        self.assertEqual(stats["malformed_removed"], 1)
        self.assertEqual(stats["total_chars"], len(docs[0]) + len(docs[1]))
        self.assertEqual(stats["min_doc_length_chars"], len(docs[0]))
        self.assertEqual(stats["max_doc_length_chars"], len(docs[1]))
        self.assertEqual(stats["train_tokens"], len(train_ids))
        self.assertEqual(stats["val_tokens"], len(val_ids))
        self.assertEqual(stats["total_tokens"], len(train_ids) + len(val_ids))


class TestPrepareCorpus(unittest.TestCase):
    def test_deduplicates_across_multiple_raw_texts(self):
        shared_paragraph = "This exact paragraph appears in both source texts."
        raw_a = f"{shared_paragraph}\n\nOnly in A, and long enough to survive filtering."
        raw_b = f"{shared_paragraph}\n\nOnly in B, and also long enough to survive filtering."

        train_ids, val_ids, tokenizer, stats = prepare_corpus([raw_a, raw_b], val_split_ratio=0.34, seed=1)

        self.assertEqual(stats["duplicates_removed"], 1)
        self.assertEqual(stats["num_documents"], 3)  # shared once + A's unique + B's unique
        self.assertGreater(stats["train_tokens"] + stats["val_tokens"], 0)

    def test_real_corpus_files_have_genuine_overlap(self):
        # data/raw/experiment1_larger_corpus.txt was authored (Phase 16) by
        # extending placeholder_corpus.txt's own paragraphs verbatim -- this
        # confirms dedup finds that real overlap, not a synthetic example.
        baseline = (ROOT / "data" / "raw" / "placeholder_corpus.txt").read_text(encoding="utf-8")
        experiment1 = (ROOT / "data" / "raw" / "experiment1_larger_corpus.txt").read_text(encoding="utf-8")

        _, _, _, stats = prepare_corpus([baseline, experiment1], val_split_ratio=0.1, seed=1337)

        self.assertGreater(stats["duplicates_removed"], 0)


class TestTextDataset(unittest.TestCase):
    def setUp(self):
        self.ids = list(range(20))
        self.context_length = 5
        self.ds = TextDataset(self.ids, self.context_length)

    def test_length_is_num_valid_windows(self):
        self.assertEqual(len(self.ds), len(self.ids) - self.context_length)

    def test_target_is_input_shifted_by_one(self):
        x, y = self.ds[0]
        self.assertEqual(x.tolist(), self.ids[0:5])
        self.assertEqual(y.tolist(), self.ids[1:6])

    def test_shapes_and_dtype(self):
        x, y = self.ds[3]
        self.assertEqual(x.shape, (self.context_length,))
        self.assertEqual(y.shape, (self.context_length,))
        self.assertEqual(str(x.dtype), "torch.int64")

    def test_rejects_corpus_too_short(self):
        with self.assertRaises(ValueError):
            TextDataset([1, 2, 3], context_length=5)


class TestDataloaders(unittest.TestCase):
    def test_batches_have_expected_shape(self):
        train_ids = list(range(500))
        val_ids = list(range(500, 600))
        train_loader, val_loader = make_dataloaders(
            train_ids, val_ids, context_length=8, batch_size=16
        )
        x, y = next(iter(train_loader))
        self.assertEqual(x.shape, (16, 8))
        self.assertEqual(y.shape, (16, 8))
        self.assertEqual(len(val_loader.dataset), len(val_ids) - 8)


class TestTokenizerBuildsFromCorpus(unittest.TestCase):
    def test_round_trip_on_cleaned_corpus(self):
        raw = "Hello,   world!\r\n\r\n\r\nThis is a test."
        cleaned = clean_text(raw)
        tok = build_tokenizer(cleaned)
        self.assertEqual(tok.decode(tok.encode(cleaned)), cleaned)


if __name__ == "__main__":
    unittest.main()
