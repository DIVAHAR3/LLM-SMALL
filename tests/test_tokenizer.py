import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenizer.char_tokenizer import CharTokenizer


class TestCharTokenizer(unittest.TestCase):
    def setUp(self):
        self.corpus = "Hello world! This is a tiny sample corpus for the char-level tokenizer."
        self.tok = CharTokenizer.from_text(self.corpus)

    def test_round_trip_hello_world(self):
        text = "Hello world"
        self.assertEqual(self.tok.decode(self.tok.encode(text)), text)

    def test_vocab_size_includes_special_tokens(self):
        unique_chars = set(self.corpus)
        self.assertEqual(self.tok.vocab_size, len(unique_chars) + len(CharTokenizer.SPECIAL_TOKENS))

    def test_unknown_character_maps_to_unk(self):
        unk_id = self.tok.char_to_id[CharTokenizer.UNK_TOKEN]
        ids = self.tok.encode("日本語")  # characters absent from the corpus
        self.assertTrue(all(i == unk_id for i in ids))

    def test_special_tokens_added_and_stripped(self):
        text = "Hello"
        ids = self.tok.encode(text, add_special_tokens=True)
        self.assertEqual(ids[0], self.tok.char_to_id[CharTokenizer.BOS_TOKEN])
        self.assertEqual(ids[-1], self.tok.char_to_id[CharTokenizer.EOS_TOKEN])
        self.assertEqual(self.tok.decode(ids), text)  # skip_special_tokens defaults True

    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "tokenizer.json")
            self.tok.save(path)
            loaded = CharTokenizer.load(path)
            self.assertEqual(loaded.vocab_size, self.tok.vocab_size)
            text = "Hello world"
            self.assertEqual(loaded.decode(loaded.encode(text)), text)


class TestWithAdditionalSpecialTokens(unittest.TestCase):
    def setUp(self):
        self.corpus = "hello world"
        self.tok = CharTokenizer.from_text(self.corpus)

    def test_new_tokens_appended_at_the_end_not_inserted(self):
        original_ids = dict(self.tok.char_to_id)
        extended = self.tok.with_additional_special_tokens(["<A>", "<B>"])

        # every original token keeps its EXACT original id -- this is the
        # whole point: a pretrained embedding table indexed by these ids
        # must stay valid, which inserting in the middle would break.
        for token, original_id in original_ids.items():
            self.assertEqual(extended.char_to_id[token], original_id)

        self.assertEqual(extended.char_to_id["<A>"], self.tok.vocab_size)
        self.assertEqual(extended.char_to_id["<B>"], self.tok.vocab_size + 1)
        self.assertEqual(extended.vocab_size, self.tok.vocab_size + 2)

    def test_original_tokenizer_is_not_mutated(self):
        original_vocab_size = self.tok.vocab_size
        self.tok.with_additional_special_tokens(["<A>"])
        self.assertEqual(self.tok.vocab_size, original_vocab_size)
        self.assertNotIn("<A>", self.tok.char_to_id)

    def test_new_tokens_encode_and_decode_correctly(self):
        extended = self.tok.with_additional_special_tokens(["<A>"])
        a_id = extended.char_to_id["<A>"]
        text_ids = extended.encode("hello")
        full_ids = [a_id] + text_ids
        self.assertEqual(extended.decode(full_ids, skip_special_tokens=False), "<A>hello")

    def test_rejects_a_token_already_in_the_vocabulary(self):
        with self.assertRaises(ValueError):
            self.tok.with_additional_special_tokens([CharTokenizer.BOS_TOKEN])


if __name__ == "__main__":
    unittest.main()
