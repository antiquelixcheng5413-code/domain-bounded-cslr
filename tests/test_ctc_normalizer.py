import tempfile
import unittest
from pathlib import Path

import numpy as np

from cslr.contracts import SampleRecord
from cslr.recognition.dataset import (
    FeatureNormalizer,
    GlossSequenceDataset,
    build_normalizer,
    feature_view_indices,
)
from cslr.recognition.gloss_sequence import build_ordered_vocabulary


def _record(sample_id: str, label: str = "我/要") -> SampleRecord:
    return SampleRecord(sample_id, Path(f"{sample_id}.mp4"), label, "A", "s", "train")


def _save(root: Path, sample_id: str, values: np.ndarray) -> None:
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / f"{sample_id}.npy", values.astype(np.float32))


class FeatureNormalizerTests(unittest.TestCase):
    def test_fit_computes_per_dimension_statistics(self) -> None:
        first = np.asarray([[1.0, 10.0], [3.0, 20.0]])
        second = np.asarray([[5.0, 30.0], [7.0, 40.0]])
        normalizer = FeatureNormalizer.fit([first, second])
        self.assertAlmostEqual(normalizer.mean[0], 4.0)
        self.assertAlmostEqual(normalizer.mean[1], 25.0)
        self.assertAlmostEqual(normalizer.std[0], np.std([1, 3, 5, 7]))

    def test_apply_standardises_to_zero_mean_unit_scale(self) -> None:
        features = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        normalizer = FeatureNormalizer.fit([features])
        transformed = normalizer.apply(features)
        self.assertAlmostEqual(float(transformed.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(transformed.std()), 1.0, places=5)

    def test_constant_dimension_is_not_divided_by_zero(self) -> None:
        features = np.asarray([[2.0, 5.0], [2.0, 7.0]])
        normalizer = FeatureNormalizer.fit([features])
        transformed = normalizer.apply(features)
        self.assertTrue(np.isfinite(transformed).all())
        self.assertTrue(np.allclose(transformed[:, 0], 0.0))

    def test_width_mismatch_is_rejected(self) -> None:
        normalizer = FeatureNormalizer.fit([np.zeros((4, 3), dtype=np.float32)])
        with self.assertRaises(ValueError):
            normalizer.apply(np.zeros((4, 5), dtype=np.float32))

    def test_dict_round_trip(self) -> None:
        normalizer = FeatureNormalizer.fit([np.asarray([[1.0, 2.0], [3.0, 4.0]])])
        restored = FeatureNormalizer.from_dict(normalizer.as_dict())
        self.assertEqual(restored.mean, normalizer.mean)
        self.assertEqual(restored.std, normalizer.std)

    def test_fit_rejects_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            FeatureNormalizer.fit([])

    def test_dataset_applies_the_normalizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = np.asarray([[1.0, 1.0], [3.0, 3.0]], dtype=np.float32)
            _save(root, "s1", raw)
            vocabulary, _ = build_ordered_vocabulary(["我/要"], min_frequency=1)
            normalizer = FeatureNormalizer.fit([raw])
            plain = GlossSequenceDataset([_record("s1")], root, vocabulary)
            scaled = GlossSequenceDataset([_record("s1")], root, vocabulary, normalizer)
            self.assertTrue(np.allclose(plain[0].features, raw))
            self.assertFalse(np.allclose(scaled[0].features, raw))
            self.assertAlmostEqual(float(scaled[0].features.mean()), 0.0, places=5)

    def test_build_normalizer_uses_only_the_given_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _save(root, "a", np.asarray([[0.0, 0.0]], dtype=np.float32))
            _save(root, "b", np.asarray([[100.0, 100.0]], dtype=np.float32))
            normalizer = build_normalizer([_record("a")], root)
            self.assertAlmostEqual(normalizer.mean[0], 0.0)


class FeatureViewTests(unittest.TestCase):
    def test_full_view_selects_every_dimension(self) -> None:
        self.assertIsNone(feature_view_indices("full"))

    def test_hands_view_selects_the_hand_block(self) -> None:
        columns = feature_view_indices("hands")
        self.assertEqual(columns, list(range(0, 126)))

    def test_combined_view_concatenates_blocks(self) -> None:
        columns = feature_view_indices("hands+hand_deltas")
        self.assertEqual(len(columns), 252)
        self.assertEqual(columns[:3], [0, 1, 2])
        self.assertIn(186, columns)

    def test_unknown_block_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            feature_view_indices("tentacles")

    def test_empty_view_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            feature_view_indices("  ")

    def test_dataset_applies_the_view_before_normalising(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = np.arange(96 * 368, dtype=np.float32).reshape(96, 368)
            _save(root, "s1", raw)
            vocabulary, _ = build_ordered_vocabulary(["我/要"], min_frequency=1)
            dataset = GlossSequenceDataset(
                [_record("s1")], root, vocabulary, feature_view="hands+hand_deltas"
            )
            sample = dataset[0]
            self.assertEqual(sample.features.shape, (96, 252))
            self.assertTrue(np.allclose(sample.features[0], np.concatenate([raw[0, :126], raw[0, 186:312]])))

    def test_build_normalizer_matches_the_view_width(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _save(root, "a", np.ones((4, 368), dtype=np.float32))
            normalizer = build_normalizer([_record("a")], root, feature_view="hands")
            self.assertEqual(len(normalizer.mean), 126)


if __name__ == "__main__":
    unittest.main()
