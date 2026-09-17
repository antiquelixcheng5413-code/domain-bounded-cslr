"""Unit tests for Part 3 Chinese text normalization and tokenization."""

from __future__ import annotations

import unittest

from cslr.translation.text import (
    TranslationTextNormalizer,
    build_vocab,
    character_tokenize,
)


class TranslationTextNormalizerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.n = TranslationTextNormalizer()

    def test_zero_width_removed(self) -> None:
        raw = "你\u200b好\u200d"
        out = self.n(raw)
        self.assertEqual(out.normalized, "你好")
        self.assertTrue(out.changed)

    def test_whitespace_collapsed(self) -> None:
        raw = "2023 年\t高考  到了"
        out = self.n(raw)
        self.assertEqual(out.normalized, "2023 年 高考 到了")

    def test_punctuation_and_words_untouched(self) -> None:
        raw = "你好，世界！"
        out = self.n(raw)
        self.assertEqual(out.normalized, raw)
        self.assertFalse(out.changed)

    def test_changed_flag(self) -> None:
        self.assertFalse(self.n("正常句子。").changed)
        self.assertTrue(self.n("正常\u200b句子。").changed)

    def test_empty(self) -> None:
        self.assertEqual(self.n("").normalized, "")


class CharacterTokenizeTest(unittest.TestCase):
    def test_cjk_separated(self) -> None:
        self.assertEqual(character_tokenize("高考到了"), ["高", "考", "到", "了"])

    def test_ascii_run_compact(self) -> None:
        self.assertEqual(character_tokenize("2023年"), ["2023", "年"])

    def test_mixed(self) -> None:
        self.assertEqual(character_tokenize("40平米"), ["40", "平", "米"])

    def test_whitespace_skipped(self) -> None:
        self.assertEqual(character_tokenize("高 考"), ["高", "考"])


class BuildVocabTest(unittest.TestCase):
    def test_stable_order(self) -> None:
        vocab = build_vocab(["高考", "考试"])
        self.assertEqual(vocab["<pad>"], 0)
        self.assertEqual(vocab["<bos>"], 1)
        self.assertEqual(vocab["<eos>"], 2)
        self.assertEqual(vocab["<unk>"], 3)
        self.assertIn("高", vocab)
        self.assertIn("考", vocab)

    def test_no_duplicates(self) -> None:
        vocab = build_vocab(["嘿嘿", "哈哈"])
        self.assertEqual(vocab["嘿"], vocab["嘿"])


if __name__ == "__main__":
    unittest.main()