import unittest

from cslr.recognition.gloss_sequence import UNKNOWN_TOKEN
from scripts.gloss_split_wer import FUNCTION_WORDS, filter_subsequence, subset_stats

PUNCT = {"。"}


class FilterSubsequenceTests(unittest.TestCase):
    def test_keeps_only_target_tokens(self) -> None:
        tokens = ["我", "了", "房子", "。", UNKNOWN_TOKEN]
        self.assertEqual(filter_subsequence(tokens, frozenset({"我", "房子"})), ["我", "房子"])

    def test_drops_unknown_marker_always(self) -> None:
        self.assertEqual(filter_subsequence([UNKNOWN_TOKEN, "房子"], frozenset({"房子"})), ["房子"])

    def test_empty_target_yields_empty(self) -> None:
        self.assertEqual(filter_subsequence(["了", "又"], frozenset()), [])


class SubsetStatsTests(unittest.TestCase):
    def test_ignores_samples_without_target_reference(self) -> None:
        references = [["房子", "了"], ["只", "有", "标点", "。"]]
        predictions = [["房子"], ["只", "有", "标点", "。"]]
        stats = subset_stats(references, predictions, frozenset({"房子"}))
        self.assertEqual(stats["reference_tokens"], 1)
        self.assertEqual(stats["covered_samples"], 1)
        self.assertEqual(stats["wer"], 0.0)

    def test_wer_counts_errors_on_subsequence(self) -> None:
        references = [["房子", "了", "休息"], ["我", "房子"]]
        predictions = [["房子", "休息"], ["我"]]
        stats = subset_stats(references, predictions, frozenset({"房子", "休息", "我"}))
        # sample1: ref [房子,休息] vs pred [房子,休息] -> 0 err over 2; sample2: ref [我,房子] vs pred [我] -> 1 over 2
        self.assertEqual(stats["wer"], 1 / 4)
        # matched tokens: 房子x1 休息x1 我x1 -> 3 matched of 4 reference tokens
        self.assertEqual(stats["token_recall"], 0.75)

    def test_recall_counts_only_matching_occurrences(self) -> None:
        references = [["房子", "房子"]]
        predictions = [["房子"]]
        stats = subset_stats(references, predictions, frozenset({"房子"}))
        self.assertEqual(stats["reference_tokens"], 2)
        self.assertEqual(stats["matched_tokens"], 1)
        self.assertEqual(stats["token_recall"], 0.5)

    def test_no_covered_samples_yields_none_rates(self) -> None:
        references = [["了"], ["又"]]
        predictions = [["房子"], ["休息"]]
        stats = subset_stats(references, predictions, frozenset({"房子", "休息"}))
        self.assertEqual(stats["covered_samples"], 0)
        self.assertIsNone(stats["wer"])
        self.assertIsNone(stats["token_recall"])
        self.assertEqual(stats["reference_tokens"], 0)


class FunctionWordSetTests(unittest.TestCase):
    def test_core_function_words_are_members(self) -> None:
        for token in ("了", "的", "又", "也", "都", "就", "在", "把", "被", "和", "与", "或",
                      "吗", "呢", "吧", "不", "没", "很", "更", "最"):
            self.assertIn(token, FUNCTION_WORDS)

    def test_content_words_are_not_members(self) -> None:
        for token in ("房子", "休息", "行李", "告诉", "学校", "卖", "票", "天气"):
            self.assertNotIn(token, FUNCTION_WORDS)


if __name__ == "__main__":
    unittest.main()
