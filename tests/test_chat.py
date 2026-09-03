import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.chat import chat
from model.gpt import GPTModel
from tokenizer.char_tokenizer import CharTokenizer
from training.chat_format import ASSISTANT_TOKEN, ROLE_TOKENS, SYSTEM_TOKEN, USER_TOKEN, build_example_ids, build_prompt_ids


def make_chat_tokenizer():
    base = CharTokenizer.from_text("hello world how are you today")
    return base.with_additional_special_tokens(ROLE_TOKENS)


class TestBuildExampleIds(unittest.TestCase):
    def setUp(self):
        self.tok = make_chat_tokenizer()

    def test_structure_matches_role_token_content_role_token_content_pattern(self):
        ids = build_example_ids(self.tok, "hello", "how are you", "today")
        expected = (
            [self.tok.char_to_id[SYSTEM_TOKEN]] + self.tok.encode("hello")
            + [self.tok.char_to_id[USER_TOKEN]] + self.tok.encode("how are you")
            + [self.tok.char_to_id[ASSISTANT_TOKEN]] + self.tok.encode("today")
        )
        self.assertEqual(ids, expected)

    def test_prompt_ids_end_exactly_at_the_assistant_token(self):
        ids = build_prompt_ids(self.tok, "hello", "how are you")
        self.assertEqual(ids[-1], self.tok.char_to_id[ASSISTANT_TOKEN])
        # nothing after <ASSISTANT> -- inference fills that in
        full = build_example_ids(self.tok, "hello", "how are you", "today")
        self.assertEqual(ids, full[: len(ids)])


class _AlwaysPredictModel(nn.Module):
    """Deterministically predicts one fixed token id regardless of input,
    matching the test double pattern in tests/test_generate.py."""

    def __init__(self, vocab_size, context_length, favored_id):
        super().__init__()
        self.config = {"context_length": context_length}
        self.vocab_size = vocab_size
        self.favored_id = favored_id

    def forward(self, token_ids):
        batch_size, seq_len = token_ids.shape
        logits = torch.full((batch_size, seq_len, self.vocab_size), -100.0)
        logits[:, :, self.favored_id] = 100.0
        return logits


class TestChat(unittest.TestCase):
    def setUp(self):
        self.tok = make_chat_tokenizer()

    def test_returns_only_generated_text_not_the_prompt(self):
        torch.manual_seed(0)
        model = GPTModel(
            self.tok.vocab_size, context_length=32, embedding_dim=8,
            num_layers=2, num_heads=2, ffn_hidden_dim=16, dropout=0.0,
        )
        reply = chat(model, self.tok, "how are you", max_new_tokens=10, greedy=True)
        self.assertNotIn("<SYSTEM>", reply)
        self.assertNotIn("<USER>", reply)
        self.assertNotIn("<ASSISTANT>", reply)

    def test_stops_immediately_when_the_model_only_wants_to_start_a_new_turn(self):
        user_id = self.tok.char_to_id[USER_TOKEN]
        model = _AlwaysPredictModel(self.tok.vocab_size, context_length=32, favored_id=user_id)
        reply = chat(model, self.tok, "how are you", max_new_tokens=20, greedy=True)
        self.assertEqual(reply, "")  # the very first generated token was the halt signal

    def test_hallucinated_second_assistant_token_also_halts_and_never_leaks_into_the_reply(self):
        # Regression test: a real run once leaked literal "<ASSISTANT>" text
        # into displayed replies, because only SYSTEM/USER were stop tokens
        # -- a malformed back-to-back <ASSISTANT> was never caught. This
        # directly, deterministically exercises that exact path.
        assistant_id = self.tok.char_to_id[ASSISTANT_TOKEN]
        model = _AlwaysPredictModel(self.tok.vocab_size, context_length=32, favored_id=assistant_id)
        reply = chat(model, self.tok, "how are you", max_new_tokens=20, greedy=True)
        self.assertEqual(reply, "")
        self.assertNotIn("<ASSISTANT>", reply)

    def test_hallucinated_next_turn_is_not_included_in_the_reply(self):
        # favored token is a real vocab character ("o"); since generation
        # never emits SYSTEM/USER, it should run the full max_new_tokens
        # and the reply should be exactly that many "o"s.
        o_id = self.tok.char_to_id["o"]
        model = _AlwaysPredictModel(self.tok.vocab_size, context_length=32, favored_id=o_id)
        reply = chat(model, self.tok, "how are you", max_new_tokens=15, greedy=True)
        self.assertEqual(reply, "o" * 15)


if __name__ == "__main__":
    unittest.main()
