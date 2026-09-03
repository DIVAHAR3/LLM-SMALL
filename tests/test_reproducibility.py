import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.reproducibility import (
    capture_run_metadata,
    get_dataset_info,
    get_environment_info,
    get_git_commit,
    set_seed,
)


class TestSetSeed(unittest.TestCase):
    def test_same_seed_produces_identical_torch_random_draws(self):
        set_seed(42)
        first = torch.rand(5)
        set_seed(42)
        second = torch.rand(5)
        self.assertTrue(torch.equal(first, second))

    def test_different_seeds_produce_different_draws(self):
        set_seed(1)
        first = torch.rand(5)
        set_seed(2)
        second = torch.rand(5)
        self.assertFalse(torch.equal(first, second))

    def test_seeds_python_random_module_too(self):
        import random
        set_seed(7)
        first = [random.random() for _ in range(3)]
        set_seed(7)
        second = [random.random() for _ in range(3)]
        self.assertEqual(first, second)

    def test_seeding_makes_a_freshly_constructed_models_weights_reproducible(self):
        # the actual point of set_seed() -- weight init draws from the RNG
        # the moment a layer is constructed, so this is the real proof
        import torch.nn as nn
        set_seed(123)
        model_a = nn.Linear(8, 4)
        set_seed(123)
        model_b = nn.Linear(8, 4)
        self.assertTrue(torch.equal(model_a.weight, model_b.weight))


class TestGetGitCommit(unittest.TestCase):
    def test_returns_a_plausible_commit_hash_and_boolean_dirty_flag(self):
        get_git_commit.cache_clear()
        commit, dirty = get_git_commit()
        # this project IS a real git checkout, so a real result is expected --
        # not mocking here specifically to prove it actually queries git
        self.assertIsInstance(commit, str)
        self.assertEqual(len(commit), 40)  # full SHA-1 hex length
        self.assertIsInstance(dirty, bool)

    def test_result_is_cached_not_requeried_every_call(self):
        get_git_commit.cache_clear()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = "abc123\n"
            mock_run.return_value.check_returncode = lambda: None
            get_git_commit()
            get_git_commit()
            get_git_commit()
        # 2 subprocess calls (rev-parse + status) for the FIRST call only,
        # not 6 for three calls -- this is what keeps save_checkpoint() fast
        self.assertEqual(mock_run.call_count, 2)
        get_git_commit.cache_clear()

    def test_returns_none_none_when_git_is_unavailable(self):
        get_git_commit.cache_clear()
        with patch("subprocess.run", side_effect=FileNotFoundError("no git")):
            commit, dirty = get_git_commit()
        self.assertIsNone(commit)
        self.assertIsNone(dirty)
        get_git_commit.cache_clear()


class TestGetEnvironmentInfo(unittest.TestCase):
    def test_contains_the_expected_keys_with_sane_values(self):
        info = get_environment_info()
        self.assertIn(".", info["python_version"])
        self.assertEqual(info["torch_version"], torch.__version__)
        self.assertIsInstance(info["cpu_count"], int)
        self.assertIn(info["device"], ("cpu", "cuda"))


class TestGetDatasetInfo(unittest.TestCase):
    def test_returns_none_when_no_processed_dataset_exists(self):
        with patch("training.reproducibility.DATASET_META_PATH", Path("no/such/file.json")):
            self.assertIsNone(get_dataset_info())

    def test_returns_parsed_meta_json_content_when_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_path = Path(tmpdir) / "meta.json"
            meta_path.write_text(json.dumps({"source_file": "corpus.txt", "vocab_size": 51}), encoding="utf-8")
            with patch("training.reproducibility.DATASET_META_PATH", meta_path):
                info = get_dataset_info()
        self.assertEqual(info, {"source_file": "corpus.txt", "vocab_size": 51})


class TestCaptureRunMetadata(unittest.TestCase):
    def test_bundles_seed_git_environment_and_dataset_info(self):
        metadata = capture_run_metadata(seed=1337)
        self.assertEqual(metadata["seed"], 1337)
        self.assertIn("git_commit", metadata)
        self.assertIn("git_dirty", metadata)
        self.assertIn("environment", metadata)
        self.assertIn("dataset", metadata)

    def test_passes_through_a_none_seed_unchanged(self):
        # a checkpoint saved without a seed in its config shouldn't fabricate one
        metadata = capture_run_metadata(seed=None)
        self.assertIsNone(metadata["seed"])


class TestEndToEndReproducibility(unittest.TestCase):
    """The actual Phase 29 stop condition, proven directly rather than
    just asserting the metadata-recording mechanism works in isolation:
    given the same seed, an entire run -- weight init through final
    trained weights, including dropout and DataLoader shuffling -- is
    bit-for-bit reproducible from a fresh start."""

    def test_two_independent_runs_with_the_same_seed_produce_bit_identical_models(self):
        from model.gpt import GPTModel
        from torch.utils.data import DataLoader
        from training.dataset import TextDataset
        from training.train import train

        def run_once():
            set_seed(2024)  # must happen BEFORE model construction
            model = GPTModel(
                vocab_size=15, context_length=8, embedding_dim=8,
                num_layers=2, num_heads=2, ffn_hidden_dim=16, dropout=0.1,
            )
            ids = list(range(15)) * 20
            loader = DataLoader(TextDataset(ids, context_length=8), batch_size=4, shuffle=True)
            config = {
                "learning_rate": 1e-2, "weight_decay": 0.0, "epochs": 1,
                "grad_accumulation_steps": 1, "grad_clip_norm": None, "seed": 2024,
            }
            train(model, loader, loader, config, max_steps=5, log_fn=lambda msg: None)
            return model

        model_a = run_once()
        model_b = run_once()

        for (name_a, param_a), (name_b, param_b) in zip(model_a.named_parameters(), model_b.named_parameters()):
            self.assertEqual(name_a, name_b)
            self.assertTrue(torch.equal(param_a, param_b), f"{name_a} diverged between two seeded runs")

    def test_different_seeds_produce_different_trained_models(self):
        # the negative case -- proves the equality above isn't a trivial
        # "everything is always equal regardless of seed" false positive
        from model.gpt import GPTModel
        from torch.utils.data import DataLoader
        from training.dataset import TextDataset
        from training.train import train

        def run_once(seed):
            set_seed(seed)
            model = GPTModel(
                vocab_size=15, context_length=8, embedding_dim=8,
                num_layers=2, num_heads=2, ffn_hidden_dim=16, dropout=0.1,
            )
            ids = list(range(15)) * 20
            loader = DataLoader(TextDataset(ids, context_length=8), batch_size=4, shuffle=True)
            config = {
                "learning_rate": 1e-2, "weight_decay": 0.0, "epochs": 1,
                "grad_accumulation_steps": 1, "grad_clip_norm": None, "seed": seed,
            }
            train(model, loader, loader, config, max_steps=5, log_fn=lambda msg: None)
            return model

        model_a = run_once(1)
        model_b = run_once(2)

        any_different = any(
            not torch.equal(p1, p2) for p1, p2 in zip(model_a.parameters(), model_b.parameters())
        )
        self.assertTrue(any_different)


if __name__ == "__main__":
    unittest.main()
