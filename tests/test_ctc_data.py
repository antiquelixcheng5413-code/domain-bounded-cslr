import tempfile
import unittest
from pathlib import Path

import numpy as np

from cslr.contracts import SampleRecord
from cslr.data.ctc import CTCDataset, CTCVocabulary, collate_ctc_samples


class CTCDataTests(unittest.TestCase):
    def test_dataset_preserves_target_order_and_batch_lengths(self) -> None:
        record = SampleRecord("s1", Path("a.mp4"), "B/A/B", "signer", "session", "train")
        vocabulary = CTCVocabulary(tokens=["A", "B"], blank_index=0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = np.ones((48, 368), dtype=np.float32)
            features[-1] = 0.0
            np.save(root / "s1.npy", features)
            dataset = CTCDataset([record], root, vocabulary)
            sample = dataset[0]
            batch = collate_ctc_samples([sample])

        self.assertEqual(sample.targets.tolist(), [2, 1, 2])
        self.assertEqual(sample.input_length, 47)
        self.assertEqual(batch["targets"].tolist(), [2, 1, 2])
        self.assertEqual(batch["target_lengths"].tolist(), [3])

    def test_vocabulary_rejects_blank_outside_model_classes(self) -> None:
        with self.assertRaises(ValueError):
            CTCVocabulary(tokens=["A", "B"], blank_index=3)
