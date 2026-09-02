import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tokenizer.bpe_tokenizer import BPETokenizer


class TestTraining(unittest.TestCase):
    def test_most_frequent_pair_is_merged_first(self):
        # "aa" appears far more often (in "aaaa" x3) than any other pair.
        text = "aaaa aaaa aaaa bbbb"
        base_vocab = len(BPETokenizer.SPECIAL_TOKENS) + len(set(text))
        tok = BPETokenizer.from_text(text, vocab_size=base_vocab + 1)
        self.assertEqual(tok.merges[0], ("a", "a"))

    def test_degrades_to_character_level_when_vocab_size_too_small(self):
        text = "hello world"
        base_vocab = len(BPETokenizer.SPECIAL_TOKENS) + len(set(text))
        tok = BPETokenizer.from_text(text, vocab_size=base_vocab)
        self.assertEqual(tok.merges, [])
        self.assertEqual(tok.vocab_size, base_vocab)

    def test_vocab_size_is_respected_when_enough_merges_are_available(self):
        # Needs enough distinct words/pairs to reach the target -- a small
        # repeated-word corpus runs out of mergeable pairs once every word
        # chunk collapses to a single token (see the "stops early" test
        # below), so this uses the real, varied project corpus instead.
        text = (ROOT / "data" / "raw" / "placeholder_corpus.txt").read_text(encoding="utf-8")
        tok = BPETokenizer.from_text(text, vocab_size=200)
        self.assertEqual(tok.vocab_size, 200)

    def test_stops_early_once_every_word_chunk_is_a_single_token(self):
        # A small, repetitive corpus has a hard ceiling on achievable vocab
        # size: once each distinct word/whitespace chunk fully collapses
        # into one token, it stops contributing pairs -- no rule can push
        # past that ceiling, however large vocab_size is asked for.
        text = "the cat sat on the mat. the cat ate the rat. " * 5
        tok = BPETokenizer.from_text(text, vocab_size=80)
        self.assertLess(tok.vocab_size, 80)

    def test_rejects_vocab_size_too_small_for_special_tokens(self):
        with self.assertRaises(ValueError):
            BPETokenizer.from_text("ab", vocab_size=1)


class TestRoundTrip(unittest.TestCase):
    def test_round_trips_on_training_text(self):
        text = "the quick brown fox jumps over the lazy dog"
        tok = BPETokenizer.from_text(text, vocab_size=60)
        self.assertEqual(tok.decode(tok.encode(text)), text)

    def test_round_trips_preserving_exact_whitespace(self):
        text = "a  b\tc\nd   e"  # irregular spacing, tabs, newlines
        tok = BPETokenizer.from_text(text, vocab_size=30)
        self.assertEqual(tok.decode(tok.encode(text)), text)

    def test_unknown_character_maps_to_unk(self):
        tok = BPETokenizer.from_text("hello world", vocab_size=25)
        ids = tok.encode("hello 日本")  # contains characters absent from training
        self.assertIn(tok.token_to_id[BPETokenizer.UNK_TOKEN], ids)

    def test_special_tokens_added_and_stripped(self):
        tok = BPETokenizer.from_text("hello world", vocab_size=25)
        ids = tok.encode("hello", add_special_tokens=True)
        self.assertEqual(ids[0], tok.token_to_id[BPETokenizer.BOS_TOKEN])
        self.assertEqual(ids[-1], tok.token_to_id[BPETokenizer.EOS_TOKEN])
        self.assertEqual(tok.decode(ids), "hello")  # skip_special_tokens defaults True


class TestSaveLoad(unittest.TestCase):
    def test_save_and_load_round_trip_preserves_merges_and_behavior(self):
        text = "the cat sat on the mat. the cat ate the rat. " * 5
        tok = BPETokenizer.from_text(text, vocab_size=80)
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "bpe.json")
            tok.save(path)
            loaded = BPETokenizer.load(path)
            self.assertEqual(loaded.vocab_size, tok.vocab_size)
            self.assertEqual(loaded.merges, tok.merges)
            self.assertEqual(loaded.decode(loaded.encode(text)), text)
            self.assertEqual(loaded.encode(text), tok.encode(text))


class TestCompression(unittest.TestCase):
    def test_merges_strictly_reduce_token_count_vs_characters(self):
        text = "the cat sat on the mat. the cat ate the rat. " * 5
        tok = BPETokenizer.from_text(text, vocab_size=120)
        char_count = len([c for c in text])
        token_count = len(tok.encode(text))
        self.assertLess(token_count, char_count)


class TestEncodeAlgorithmCorrectness(unittest.TestCase):
    """Directly constructed tokenizers (bypassing from_text) to test the
    encode-time merge-replay algorithm in full isolation."""

    def test_applies_the_single_highest_priority_merge_available_each_round(self):
        # merges learned in this exact order: "a"+"b" -> "ab" (rank 0),
        # then "ab"+"c" -> "abc" (rank 1). Encoding "abc" must apply BOTH,
        # in that order, ending as one token, not stop after the first.
        token_to_id = {t: i for i, t in enumerate(BPETokenizer.SPECIAL_TOKENS + ["a", "b", "c", "ab", "abc"])}
        tok = BPETokenizer(token_to_id, merges=[("a", "b"), ("ab", "c")])
        pieces = tok._bpe_encode_word(("a", "b", "c"))
        self.assertEqual(pieces, ["abc"])

    def test_merges_every_occurrence_of_the_winning_pair_in_one_round_not_just_the_first(self):
        # only merge rule: "a"+"a" -> "aa". "aaaa" has three adjacent (a,a)
        # pairs but only two DISJOINT occurrences fit left-to-right in one
        # pass; after that "aa"+"aa" has no rule, so it stops at 2 tokens.
        token_to_id = {t: i for i, t in enumerate(BPETokenizer.SPECIAL_TOKENS + ["a", "aa"])}
        tok = BPETokenizer(token_to_id, merges=[("a", "a")])
        pieces = tok._bpe_encode_word(("a", "a", "a", "a"))
        self.assertEqual(pieces, ["aa", "aa"])

    def test_does_not_apply_a_lower_priority_merge_before_a_higher_priority_one(self):
        # both ("b","c")->"bc" and ("a","b")->"ab" are possible in "abc",
        # but ("a","b") was learned first (rank 0), so it must win.
        token_to_id = {t: i for i, t in enumerate(BPETokenizer.SPECIAL_TOKENS + ["a", "b", "c", "ab", "bc"])}
        tok = BPETokenizer(token_to_id, merges=[("a", "b"), ("b", "c")])
        pieces = tok._bpe_encode_word(("a", "b", "c"))
        self.assertEqual(pieces, ["ab", "c"])


if __name__ == "__main__":
    unittest.main()
