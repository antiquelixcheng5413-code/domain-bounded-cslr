import tempfile
import unittest
from pathlib import Path

import numpy as np

from cslr.contracts import SampleRecord
from cslr.data.feature_bundle import verify_feature_bundle


class FeatureBundleTests(unittest.TestCase):
    def test_valid_bundle_has_stable_integrity_digest(self) -> None:
        records = [
            SampleRecord("s1", Path("one.mp4"), "A", "p1", "s", "train"),
            SampleRecord("s2", Path("two.mp4"), "B", "p2", "s", "test"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "s1.npy", np.zeros((48, 368), dtype=np.float32))
            np.save(root / "s2.npy", np.ones((48, 368), dtype=np.float32))
            first = verify_feature_bundle(records, root, include_sha256=True)
            second = verify_feature_bundle(records, root, include_sha256=True)

        self.assertEqual(first["status"], "ok")
        self.assertEqual(first["valid_records"], 2)
        self.assertEqual(first["split_counts"], {"test": 1, "train": 1})
        self.assertEqual(first["integrity"], second["integrity"])

    def test_invalid_and_missing_features_are_reported(self) -> None:
        records = [
            SampleRecord("s1", Path("one.mp4"), "A", "p1", "s", "train"),
            SampleRecord("s2", Path("two.mp4"), "B", "p2", "s", "dev"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "s1.npy", np.zeros((47, 368), dtype=np.float64))
            result = verify_feature_bundle(records, root)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["missing_sample_ids"], ["s2"])
        self.assertEqual(result["invalid_shape_count"], 1)
