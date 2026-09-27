import unittest

from cslr.recognition.text_metrics import (
    bleu,
    chrf,
    distinct_ngrams,
    exact_match,
    rouge_l,
)


class BleuTests(unittest.TestCase):
    def test_perfect_corpus_scores_one(self) -> None:
        reference = [list("上海站到了")]
        scores = bleu(reference, reference)
        self.assertAlmostEqual(scores["bleu_1"], 1.0)
        self.assertAlmostEqual(scores["bleu_4"], 1.0)

    def test_bad_hypothesis_scores_lower(self) -> None:
        reference = [list("上海站到了")]
        good = bleu(reference, reference)["bleu_1"]
        bad = bleu(reference, [list("完全不同")])["bleu_1"]
        self.assertLess(bad, good)

    def test_length_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            bleu([list("a")], [])

    def test_is_deterministic(self) -> None:
        reference = [list("今天天气很好")]
        hypothesis = [list("今天天气不错")]
        first = bleu(reference, hypothesis)
        for _ in range(3):
            self.assertEqual(bleu(reference, hypothesis), first)


class RougeTests(unittest.TestCase):
    def test_identical_text_scores_one(self) -> None:
        reference = [list("我 要 挂 号".replace(" ", ""))]
        self.assertAlmostEqual(rouge_l(reference, reference), 1.0)

    def test_partial_overlap_is_between_zero_and_one(self) -> None:
        score = rouge_l([list("上海站")], [list("上海")])
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_empty_hypothesis_scores_zero(self) -> None:
        self.assertAlmostEqual(rouge_l([list("上海")], [[]]), 0.0)


class ChrfTests(unittest.TestCase):
    def test_identical_text_scores_one(self) -> None:
        reference = [list("医院前台")]
        self.assertAlmostEqual(chrf(reference, reference), 1.0)

    def test_wrong_text_scores_lower(self) -> None:
        reference = [list("医院前台")]
        self.assertLess(chrf(reference, [list("完全无关内容")]), 1.0)

    def test_empty_pair_is_zero(self) -> None:
        self.assertAlmostEqual(chrf([[]], [[]]), 0.0)


class ExactMatchTests(unittest.TestCase):
    def test_counts_only_identical_strings(self) -> None:
        self.assertAlmostEqual(exact_match(["a", "b"], ["a", "c"]), 0.5)


class DistinctTests(unittest.TestCase):
    def test_collapse_is_visible(self) -> None:
        collapsed = distinct_ngrams([list("我 要"), list("我 要")], order=1)
        self.assertEqual(collapsed["unique"], 3)  # 我, 要, and the space character
        self.assertLess(float(collapsed["distinct"]), 1.0)

    def test_empty_corpus(self) -> None:
        self.assertEqual(distinct_ngrams([], order=1)["total"], 0)


if __name__ == "__main__":
    unittest.main()
