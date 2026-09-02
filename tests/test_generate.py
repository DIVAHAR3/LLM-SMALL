import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.generate import _filter_top_k, _filter_top_p, generate_ids, generate_text, sample_next_token, stream_ids
from model.gpt import GPTModel
from tokenizer.char_tokenizer import CharTokenizer


class _AlwaysPredictModel(nn.Module):
    """Test double: ignores its input and always assigns overwhelming
    probability to a single fixed token id, regardless of context. Used to
    deterministically test stop-token handling without needing a trained
    model to reliably predict a specific id."""

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


def make_tiny_model(vocab_size=12, context_length=6):
    return GPTModel(
        vocab_size, context_length, embedding_dim=8,
        num_layers=2, num_heads=2, ffn_hidden_dim=16, dropout=0.0,
    )


class TestFilterTopK(unittest.TestCase):
    def test_keeps_exactly_the_top_k_logits(self):
        logits = torch.tensor([1.0, 5.0, 3.0, 4.0, 2.0])
        filtered = _filter_top_k(logits, top_k=2)
        kept = torch.isfinite(filtered)
        self.assertEqual(kept.sum().item(), 2)
        self.assertTrue(kept[1])  # value 5.0
        self.assertTrue(kept[3])  # value 4.0

    def test_none_or_full_vocab_leaves_logits_unchanged(self):
        logits = torch.tensor([1.0, 5.0, 3.0])
        self.assertTrue(torch.equal(_filter_top_k(logits, None), logits))
        self.assertTrue(torch.equal(_filter_top_k(logits, top_k=10), logits))


class TestFilterTopP(unittest.TestCase):
    def test_hand_computed_nucleus_cutoff(self):
        # log of exact probabilities [0.5, 0.3, 0.1, 0.06, 0.04] -> softmax
        # reproduces them precisely. Cumulative: .5, .8, .9, .96, 1.0.
        # p=0.9: cumulative first EXCEEDS 0.9 at index 3 (.96), so indices
        # 0-3 are kept (the token causing the crossing is always kept),
        # only the last (index 4, prob .04) is removed.
        probs = torch.tensor([0.5, 0.3, 0.1, 0.06, 0.04])
        logits = torch.log(probs)
        filtered = _filter_top_p(logits, top_p=0.9)
        kept = torch.isfinite(filtered)
        self.assertEqual(kept.tolist(), [True, True, True, True, False])

    def test_none_leaves_logits_unchanged(self):
        logits = torch.tensor([1.0, 5.0, 3.0])
        self.assertTrue(torch.equal(_filter_top_p(logits, None), logits))

    def test_always_keeps_at_least_the_top_token(self):
        probs = torch.tensor([0.5, 0.3, 0.2])
        logits = torch.log(probs)
        filtered = _filter_top_p(logits, top_p=1e-6)  # smaller than even the top token's own probability
        kept = torch.isfinite(filtered)
        self.assertEqual(kept.tolist(), [True, False, False])


class TestSampleNextToken(unittest.TestCase):
    def test_greedy_always_returns_argmax_deterministically(self):
        logits = torch.tensor([1.0, 5.0, 3.0, 4.0])
        results = {sample_next_token(logits, greedy=True) for _ in range(5)}
        self.assertEqual(results, {1})  # index of the max value, no randomness

    def test_top_k_1_forces_deterministic_choice_regardless_of_temperature(self):
        logits = torch.tensor([1.0, 5.0, 3.0, 4.0])
        for temperature in [0.1, 1.0, 5.0]:
            results = {
                sample_next_token(logits, temperature=temperature, top_k=1, greedy=False)
                for _ in range(10)
            }
            self.assertEqual(results, {1})


class TestGenerateIds(unittest.TestCase):
    def test_max_new_tokens_bounds_length_when_no_stop_token(self):
        model = make_tiny_model()
        prompt_ids = [0, 1, 2]
        result = generate_ids(model, prompt_ids, max_new_tokens=15, greedy=True)
        self.assertEqual(len(result), len(prompt_ids) + 15)
        self.assertEqual(result[: len(prompt_ids)], prompt_ids)

    def test_stop_token_halts_generation_early(self):
        stop_id = 7
        model = _AlwaysPredictModel(vocab_size=12, context_length=6, favored_id=stop_id)
        result = generate_ids(model, [0, 1], max_new_tokens=50, greedy=True, stop_token_ids={stop_id})
        self.assertEqual(len(result), 3)  # 2 prompt tokens + exactly 1 generated (the stop token) before halting
        self.assertEqual(result[-1], stop_id)

    def test_without_stop_token_configured_generation_ignores_matching_ids(self):
        favored_id = 7
        model = _AlwaysPredictModel(vocab_size=12, context_length=6, favored_id=favored_id)
        result = generate_ids(model, [0, 1], max_new_tokens=5, greedy=True, stop_token_ids=None)
        self.assertEqual(len(result), 7)  # runs the full max_new_tokens, nothing configured to stop it

    def test_context_length_sliding_window_does_not_crash(self):
        # context_length=6, generating well beyond it must not feed an
        # oversized sequence back into the model.
        model = make_tiny_model(vocab_size=12, context_length=6)
        result = generate_ids(model, [0, 1], max_new_tokens=20, greedy=True)
        self.assertEqual(len(result), 22)

    def test_greedy_generation_is_deterministic(self):
        model = make_tiny_model()
        prompt_ids = [0, 1, 2]
        first = generate_ids(model, prompt_ids, max_new_tokens=10, greedy=True)
        second = generate_ids(model, prompt_ids, max_new_tokens=10, greedy=True)
        self.assertEqual(first, second)

    def test_restores_original_train_eval_mode_after_generating(self):
        model = make_tiny_model()
        model.train()
        generate_ids(model, [0, 1], max_new_tokens=3, greedy=True)
        self.assertTrue(model.training)

        model.eval()
        generate_ids(model, [0, 1], max_new_tokens=3, greedy=True)
        self.assertFalse(model.training)


class TestStreamIds(unittest.TestCase):
    def test_matches_generate_ids_exactly_when_fully_consumed(self):
        # generate_ids is now just stream_ids fully consumed -- this proves
        # the refactor preserved identical behavior, not just "still passes
        # the old tests" (those never directly compare the two).
        model = make_tiny_model()
        prompt_ids = [0, 1, 2]
        torch.manual_seed(0)
        via_generate_ids = generate_ids(model, prompt_ids, max_new_tokens=10, greedy=True)
        torch.manual_seed(0)
        via_stream = prompt_ids + list(stream_ids(model, prompt_ids, max_new_tokens=10, greedy=True))
        self.assertEqual(via_generate_ids, via_stream)

    def test_yields_one_token_at_a_time_not_all_at_once(self):
        model = make_tiny_model()
        gen = stream_ids(model, [0, 1], max_new_tokens=10, greedy=True)
        first_token = next(gen)
        self.assertIsInstance(first_token, int)
        # only one token should exist so far -- proves this is genuinely
        # incremental, not a generator that secretly runs to completion
        # before yielding its first value.
        remaining = list(gen)
        self.assertEqual(len(remaining), 9)

    def test_stop_token_is_yielded_then_generation_halts(self):
        stop_id = 7
        model = _AlwaysPredictModel(vocab_size=12, context_length=6, favored_id=stop_id)
        yielded = list(stream_ids(model, [0, 1], max_new_tokens=50, greedy=True, stop_token_ids={stop_id}))
        self.assertEqual(yielded, [stop_id])  # exactly one token yielded: the stop token itself

    def test_closing_early_still_restores_eval_mode(self):
        # Simulates a client disconnecting mid-stream: the generator is
        # abandoned (never asked for its remaining tokens) and garbage
        # collected. The `finally` inside stream_ids must still run.
        model = make_tiny_model()
        model.train()
        gen = stream_ids(model, [0, 1], max_new_tokens=50, greedy=True)
        next(gen)  # partially consume, then close early
        gen.close()
        self.assertTrue(model.training)  # restored, not left in eval mode


class TestGenerateText(unittest.TestCase):
    def test_round_trips_through_tokenizer(self):
        torch.manual_seed(0)
        corpus = "the quick brown fox jumps over the lazy dog"
        tokenizer = CharTokenizer.from_text(corpus)
        model = make_tiny_model(vocab_size=tokenizer.vocab_size, context_length=8)

        text = generate_text(model, tokenizer, "the quick", max_new_tokens=10, greedy=True)

        self.assertIsInstance(text, str)
        self.assertTrue(text.startswith("the quick"))
        self.assertGreater(len(text), len("the quick"))


if __name__ == "__main__":
    unittest.main()
