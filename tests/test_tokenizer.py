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


if __name__ == "__main__":
    unittest.main()
