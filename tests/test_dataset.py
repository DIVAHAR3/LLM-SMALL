import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.data_prep import build_tokenizer, clean_text, split_ids
from training.dataset import TextDataset, make_dataloaders


class TestCleanText(unittest.TestCase):
    def test_normalizes_crlf(self):
        self.assertEqual(clean_text("a\r\nb\rc"), "a\nb\nc")

    def test_collapses_excess_blank_lines(self):
        self.assertEqual(clean_text("a\n\n\n\n\nb"), "a\n\nb")

    def test_strips_leading_trailing_whitespace(self):
        self.assertEqual(clean_text("  \n hello \n  "), "hello")


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
