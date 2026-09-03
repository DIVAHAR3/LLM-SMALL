import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.monitoring import get_available_memory_mb


class TestGetAvailableMemory(unittest.TestCase):
    def test_returns_a_positive_number_or_none(self):
        # Contract: either a real positive figure, or None if unavailable --
        # never a guess, never a crash. Same contract as the original
        # scripts/benchmark.py version this was moved from (Phase 22).
        result = get_available_memory_mb()
        self.assertTrue(result is None or result > 0)


if __name__ == "__main__":
    unittest.main()
