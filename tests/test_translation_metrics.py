"""Unit tests for the in-repo Part 3 translation metrics."""

from __future__ import annotations

import unittest

from cslr.translation.metrics import bleu, chrf, rouge_l


class BleuTest(unittest.TestCase):
    def test_exact_match_is_high(self) -> None:
        self.assertGreater(bleu("高考到了", "高考到了"), 0.9)

    def test_wrong_is_low(self) -> None:
        self.assertLess(bleu("高考到了", "回家吃饭"), 0.2)

    def test_both_empty(self) -> None:
        self.assertEqual(bleu("", ""), 1.0)

    def test_empty_hypothesis(self) -> None:
        self.assertEqual(bleu("高考", ""), 0.0)


class RougeLTest(unittest.TestCase):
    def test_exact(self) -> None:
        self.assertGreater(rouge_l("高考到了", "高考到了"), 0.9)

    def test_partial_overlap(self) -> None:
        # shared "高" and "考" -> recall/precision positive
        self.assertGreater(rouge_l("高考到了", "高考"), 0.0)

    def test_empty(self) -> None:
        self.assertEqual(rouge_l("", ""), 1.0)


class ChrFTest(unittest.TestCase):
    def test_exact(self) -> None:
        self.assertGreater(chrf("高考到了", "高考到了"), 0.9)

    def test_no_overlap(self) -> None:
        self.assertLess(chrf("高考", "回家"), 0.1)

    def test_empty(self) -> None:
        self.assertEqual(chrf("", ""), 1.0)


if __name__ == "__main__":
    unittest.main()