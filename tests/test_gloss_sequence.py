import unittest

from cslr.recognition.gloss_sequence import (
    UNKNOWN_TOKEN,
    GlossSequenceConfig,
    build_ordered_vocabulary,
    coverage_report,
    split_gloss_sequence,
    token_is_number_with_tail,
    token_is_numeric,
)


class SplitGlossSequenceTests(unittest.TestCase):
    def test_keeps_order_and_punctuation_by_default(self) -> None:
        self.assertEqual(
            split_gloss_sequence("上海/站/到/。"), ["上海", "站", "到", "。"],
        )

    def test_can_drop_punctuation(self) -> None:
        config = GlossSequenceConfig(keep_punctuation=False)
        self.assertEqual(split_gloss_sequence("上海/站/到/。", config), ["上海", "站", "到"])

    def test_strips_trailing_variant_numbering(self) -> None:
        self.assertEqual(
            split_gloss_sequence("考试1/禁止1/退1"), ["考试", "禁止", "退"],
        )

    def test_variant_numbering_can_be_kept(self) -> None:
        config = GlossSequenceConfig(strip_variant_numbering=False)
        self.assertEqual(split_gloss_sequence("考试1/不行2", config), ["考试1", "不行2"])

    def test_variant_tokens_stay_distinct_when_numbering_kept(self) -> None:
        config = GlossSequenceConfig(strip_variant_numbering=False)
        variants = split_gloss_sequence("最1/最2", config)
        merged = split_gloss_sequence("最1/最2", GlossSequenceConfig())
        self.assertEqual(variants, ["最1", "最2"])
        self.assertEqual(merged, ["最", "最"])

    def test_keeps_pure_numeric_tokens_by_default(self) -> None:
        self.assertEqual(
            split_gloss_sequence("2/0/2/3/高/考/时间/到/。"),
            ["2", "0", "2", "3", "高", "考", "时间", "到", "。"],
        )

    def test_can_drop_pure_numeric_tokens(self) -> None:
        config = GlossSequenceConfig(keep_numeric_tokens=False)
        self.assertEqual(split_gloss_sequence("2/0/2/3/高/考", config), ["高", "考"])

    def test_keeps_number_with_measure_word_intact(self) -> None:
        self.assertEqual(
            split_gloss_sequence("2个/9折/2.2亿/15分钟/10月3号"),
            ["2个", "9折", "2.2亿", "15分钟", "10月3号"],
        )

    def test_strips_bracketed_annotations(self) -> None:
        self.assertEqual(split_gloss_sequence("带1（我）"), ["带"])
        self.assertEqual(split_gloss_sequence("（你）感染2（我）"), ["感染"])

    def test_strips_brace_annotations(self) -> None:
        self.assertEqual(
            split_gloss_sequence("英国{英国词汇第1个手势动作}"), ["英国"],
        )

    def test_annotations_can_be_kept(self) -> None:
        config = GlossSequenceConfig(strip_annotations=False, strip_variant_numbering=False)
        self.assertEqual(split_gloss_sequence("带1（我）", config), ["带1（我）"])

    def test_empty_gloss_produces_empty_sequence(self) -> None:
        self.assertEqual(split_gloss_sequence(""), [])
        self.assertEqual(split_gloss_sequence("/"), [])
        self.assertEqual(split_gloss_sequence("。", GlossSequenceConfig(keep_punctuation=False)), [])

    def test_is_deterministic(self) -> None:
        raw = "2/0/2/3/高/考/时间/到/。"
        first = split_gloss_sequence(raw)
        for _ in range(5):
            self.assertEqual(split_gloss_sequence(raw), first)

    def test_token_classifiers(self) -> None:
        self.assertTrue(token_is_numeric("20"))
        self.assertFalse(token_is_numeric("2个"))
        self.assertTrue(token_is_number_with_tail("2个"))
        self.assertTrue(token_is_number_with_tail("2.2亿"))
        self.assertFalse(token_is_number_with_tail("上海"))


class OrderedVocabularyTests(unittest.TestCase):
    def test_unknown_token_occupies_index_zero(self) -> None:
        vocabulary, _ = build_ordered_vocabulary(["我/要"], min_frequency=1)
        self.assertEqual(vocabulary.tokens[0], UNKNOWN_TOKEN)
        self.assertEqual(vocabulary.index_of(UNKNOWN_TOKEN), 0)

    def test_sorted_by_frequency_then_token(self) -> None:
        vocabulary, counts = build_ordered_vocabulary(
            ["我/要/买", "我/要/票", "我/挂号"], min_frequency=2
        )
        self.assertEqual(vocabulary.tokens[1:], ("我", "要"))
        self.assertEqual(counts["我"], 3)
        self.assertEqual(counts["要"], 2)

    def test_min_frequency_filters_rare_tokens(self) -> None:
        vocabulary, _ = build_ordered_vocabulary(["我/要/买", "我/要/票"], min_frequency=2)
        self.assertIn("我", vocabulary)
        self.assertNotIn("买", vocabulary)

    def test_max_tokens_is_applied_after_sorting(self) -> None:
        vocabulary, _ = build_ordered_vocabulary(
            ["我/要/买", "我/要/票", "我/挂号"], min_frequency=1, max_tokens=2
        )
        self.assertEqual(vocabulary.size, 3)  # <unk> + 2 real tokens

    def test_encode_uses_unknown_for_oov(self) -> None:
        vocabulary, _ = build_ordered_vocabulary(["我/要", "我/要"], min_frequency=2)
        self.assertEqual(vocabulary.encode("我/不存在的词"), [vocabulary.index_of("我"), 0])

    def test_encode_preserves_order_and_repeats(self) -> None:
        vocabulary, _ = build_ordered_vocabulary(["我/你/我", "我/你/我"], min_frequency=2)
        encoded = vocabulary.encode("我/你/我")
        self.assertEqual([vocabulary.decode(encoded)], [["我", "你", "我"]])

    def test_json_round_trip(self) -> None:
        vocabulary, _ = build_ordered_vocabulary(["我/要/买", "我/要/票"], min_frequency=2)
        restored = type(vocabulary).from_json(vocabulary.to_json())
        self.assertEqual(restored.tokens, vocabulary.tokens)
        self.assertEqual(restored.counts, vocabulary.counts)
        self.assertEqual(restored.config, vocabulary.config)

    def test_variant_stripping_shrinks_vocabulary(self) -> None:
        glosses = ["考试1/高", "考试2/高", "考试1/高"]
        stripped, _ = build_ordered_vocabulary(
            glosses, min_frequency=1, config=GlossSequenceConfig(strip_variant_numbering=True)
        )
        kept, _ = build_ordered_vocabulary(
            glosses, min_frequency=1, config=GlossSequenceConfig(strip_variant_numbering=False)
        )
        self.assertIn("考试", stripped)
        self.assertNotIn("考试1", stripped)
        self.assertIn("考试1", kept)
        self.assertGreater(kept.size, stripped.size)


class CharacterTargetTests(unittest.TestCase):
    def test_char_units_split_tokens_into_characters(self) -> None:
        vocabulary, _ = build_ordered_vocabulary(["上海/站"], min_frequency=1)
        self.assertEqual(vocabulary.units("上海/站"), ["上海", "站"])
        char_vocabulary, _ = build_ordered_vocabulary(
            ["上海/站"], min_frequency=1, config=GlossSequenceConfig(target_unit="char")
        )
        self.assertEqual(char_vocabulary.units("上海/站"), ["上", "海", "站"])

    def test_char_vocabulary_uses_characters_as_classes(self) -> None:
        vocabulary, counts = build_ordered_vocabulary(
            ["上海/站", "上/海"], min_frequency=1, config=GlossSequenceConfig(target_unit="char")
        )
        self.assertIn("上", counts)
        self.assertNotIn("上海", counts)
        self.assertEqual(
            vocabulary.encode("上海"), [vocabulary.index_of("上"), vocabulary.index_of("海")]
        )

    def test_char_targets_are_longer_than_token_targets(self) -> None:
        token_vocabulary, _ = build_ordered_vocabulary(["时间/到"], min_frequency=1)
        char_vocabulary, _ = build_ordered_vocabulary(
            ["时间/到"], min_frequency=1, config=GlossSequenceConfig(target_unit="char")
        )
        self.assertGreater(
            len(char_vocabulary.encode("时间/到")), len(token_vocabulary.encode("时间/到"))
        )

    def test_invalid_target_unit_is_rejected(self) -> None:
        config = GlossSequenceConfig(target_unit="syllable")
        with self.assertRaises(ValueError):
            build_ordered_vocabulary(["我"], min_frequency=1, config=config)

    def test_coverage_report_uses_the_configured_units(self) -> None:
        vocabulary, counts = build_ordered_vocabulary(
            ["上海/站"], min_frequency=1, config=GlossSequenceConfig(target_unit="char")
        )
        report = coverage_report(["上海/站"], vocabulary, counts)
        self.assertEqual(report["token_occurrences"], 3)


class CoverageReportTests(unittest.TestCase):
    def test_reports_oov_rate_and_lengths(self) -> None:
        train = ["我/要/挂号", "我/要/买/票", "我/要/挂号"]
        vocabulary, counts = build_ordered_vocabulary(train, min_frequency=2)
        report = coverage_report(["我/要/挂号", "我/要/新词"], vocabulary, counts)
        self.assertEqual(report["samples"], 2)
        self.assertEqual(report["token_occurrences"], 6)
        self.assertEqual(report["unknown"], 1)
        self.assertAlmostEqual(float(report["oov_rate"]), 1 / 6)
        length = report["sequence_length"]
        self.assertEqual(length["max"], 3)
        self.assertEqual(length["min"], 3)
        self.assertEqual(length["median"], 3)
        self.assertEqual(report["empty_sequences"], 0)

    def test_counts_empty_sequences(self) -> None:
        vocabulary, counts = build_ordered_vocabulary(["我/要"], min_frequency=1)
        report = coverage_report(["", "我/要"], vocabulary, counts)
        self.assertEqual(report["empty_sequences"], 1)

    def test_counts_token_kinds(self) -> None:
        vocabulary, counts = build_ordered_vocabulary(["我/要"], min_frequency=1)
        report = coverage_report(["2个/9折/2/。", "我/要"], vocabulary, counts)
        self.assertEqual(report["numeric_tokens"], 1)
        self.assertEqual(report["number_with_tail_tokens"], 2)
        self.assertEqual(report["punctuation_tokens"], 1)


if __name__ == "__main__":
    unittest.main()
