import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cslr.contracts import SampleRecord
from cslr.recognition.dataset import (
    GlossSequenceDataset,
    build_vocabulary_from_records,
    collate_samples,
    describe_split,
    load_feature,
    split_records,
    vocabulary_from_json,
    vocabulary_to_json,
)
from cslr.recognition.gloss_sequence import GlossSequenceConfig, build_ordered_vocabulary


def write_features(root: Path, sample_id: str, length: int = 5, width: int = 4) -> None:
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / f"{sample_id}.npy", np.arange(length * width, dtype=np.float32).reshape(length, width))


def record(sample_id: str, label: str, split: str = "train") -> SampleRecord:
    return SampleRecord(sample_id, Path(f"video/{sample_id}.mp4"), label, "A", split, split)


class FeatureLoadingTests(unittest.TestCase):
    def test_missing_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_feature(Path(tmp) / "nope.npy")

    def test_non_finite_features_raise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.npy"
            array = np.zeros((3, 4), dtype=np.float32)
            array[1, 2] = np.nan
            np.save(path, array)
            with self.assertRaises(ValueError):
                load_feature(path)

    def test_empty_sequence_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.npy"
            np.save(path, np.zeros((0, 4), dtype=np.float32))
            with self.assertRaises(ValueError):
                load_feature(path)

    def test_one_dimensional_array_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "flat.npy"
            np.save(path, np.zeros(4, dtype=np.float32))
            with self.assertRaises(ValueError):
                load_feature(path)


class DatasetTests(unittest.TestCase):
    def test_encodes_targets_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_features(root, "s1")
            vocabulary, _ = build_ordered_vocabulary(["我/你/我"], min_frequency=1)
            dataset = GlossSequenceDataset([record("s1", "我/你/我")], root, vocabulary)
            sample = dataset[0]
            self.assertEqual(sample.tokens, ("我", "你", "我"))
            expected_ids = tuple(vocabulary.encode("我/你/我"))
            self.assertEqual(sample.token_ids, expected_ids)
            # order and repetition are preserved (not collapsed to a set)
            self.assertEqual(len(sample.token_ids), 3)
            self.assertEqual(sample.token_ids[0], sample.token_ids[2])

    def test_empty_dataset_is_rejected(self) -> None:
        vocabulary, _ = build_ordered_vocabulary(["我"], min_frequency=1)
        with self.assertRaises(ValueError):
            GlossSequenceDataset([], Path("."), vocabulary)

    def test_split_records_filters_by_split(self) -> None:
        records = [record("a", "我", "train"), record("b", "你", "validation"), record("c", "他", "test")]
        self.assertEqual([r.sample_id for r in split_records(records, "train")], ["a"])
        self.assertEqual([r.sample_id for r in split_records(records, "test")], ["c"])

    def test_build_vocabulary_uses_train_only(self) -> None:
        records = [record("a", "我/要", "train"), record("b", "独有词", "validation")]
        vocabulary, _ = build_vocabulary_from_records(records, min_frequency=1)
        self.assertIn("我", vocabulary)
        self.assertNotIn("独有词", vocabulary)


class CollateTests(unittest.TestCase):
    def test_pads_features_and_keeps_lengths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_features(root, "s1", length=5)
            write_features(root, "s2", length=3)
            vocabulary, _ = build_ordered_vocabulary(["我/你", "我"], min_frequency=1)
            dataset = GlossSequenceDataset(
                [record("s1", "我/你"), record("s2", "我")], root, vocabulary
            )
            batch = collate_samples([dataset[0], dataset[1]])
            self.assertEqual(batch["features"].shape, (2, 5, 4))
            self.assertEqual(batch["input_lengths"], [5, 3])
            self.assertEqual(batch["mask"].tolist(), [[True] * 5, [True, True, True, False, False]])
            self.assertTrue(np.all(batch["features"][1, 3:] == 0.0))
            self.assertEqual(batch["target_lengths"], [2, 1])
            self.assertEqual(len(batch["targets"]), 3)
            self.assertEqual(batch["sample_ids"], ["s1", "s2"])

    def test_mixed_feature_widths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_features(root, "s1", length=4, width=4)
            write_features(root, "s2", length=4, width=6)
            vocabulary, _ = build_ordered_vocabulary(["我"], min_frequency=1)
            dataset = GlossSequenceDataset([record("s1", "我"), record("s2", "我")], root, vocabulary)
            with self.assertRaises(ValueError):
                collate_samples([dataset[0], dataset[1]])

    def test_empty_batch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            collate_samples([])

    def test_padding_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_features(root, "s1", length=4)
            write_features(root, "s2", length=2)
            vocabulary, _ = build_ordered_vocabulary(["我"], min_frequency=1)
            dataset = GlossSequenceDataset([record("s1", "我"), record("s2", "我")], root, vocabulary)
            first = collate_samples([dataset[0], dataset[1]])
            second = collate_samples([dataset[0], dataset[1]])
            self.assertTrue(np.array_equal(first["features"], second["features"]))
            self.assertTrue(np.array_equal(first["mask"], second["mask"]))


class DescribeSplitTests(unittest.TestCase):
    def test_reports_missing_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_features(root, "s1", length=6)
            report = describe_split([record("s1", "我"), record("s2", "你")], root)
            self.assertEqual(report["records"], 2)
            self.assertEqual(report["features_present"], 1)
            self.assertEqual(report["features_missing"], 1)
            self.assertEqual(report["missing_examples"], ["s2"])
            self.assertEqual(report["sequence_lengths"]["max"], 6)
            self.assertEqual(report["feature_dimensions"], [4])


class VocabularyPersistenceTests(unittest.TestCase):
    def test_json_round_trip_preserves_config(self) -> None:
        vocabulary, _ = build_ordered_vocabulary(
            ["考试1/我"], min_frequency=1, config=GlossSequenceConfig(strip_variant_numbering=False)
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vocab.json"
            vocabulary_to_json(vocabulary, path)
            restored = vocabulary_from_json(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(restored.tokens, vocabulary.tokens)
        self.assertFalse(restored.config.strip_variant_numbering)
        self.assertIn("tokens", payload)


if __name__ == "__main__":
    unittest.main()
