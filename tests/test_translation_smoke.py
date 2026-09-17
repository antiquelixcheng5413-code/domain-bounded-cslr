"""Unit tests for the Part 3 smoke pipeline and receipts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cslr.translation.cache import write_smoke_receipt, save_feature, load_feature


class CacheTest(unittest.TestCase):
    def test_save_load_roundtrip(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = save_feature(
                root, "train-00001", "rgb", np.zeros((4, 16), dtype=np.float32),
                model_name="test", version="v0", frame_sampling="uniform",
            )
            self.assertTrue(path.exists())
            loaded = load_feature(root, "train-00001", "rgb")
            self.assertEqual(loaded.shape, (4, 16))

    def test_refuse_overwrite_landmark(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_feature(root, "x", "landmark", np.zeros((4, 368)),
                          model_name="a", version="v1", frame_sampling="u")
            with self.assertRaises(FileExistsError):
                save_feature(root, "x", "landmark", np.ones((4, 368)),
                              model_name="b", version="v2", frame_sampling="u")

    def test_hash_tamper_detected(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_feature(root, "y", "rgb", np.zeros((2, 4)),
                          model_name="c", version="v1", frame_sampling="u")
            feat = root / "y.rgb.npy"
            data = np.load(feat)
            data[0, 0] = 99.0
            np.save(feat, data)
            with self.assertRaises(ValueError):
                load_feature(root, "y", "rgb")


class ReceiptTest(unittest.TestCase):
    def test_receipt_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_smoke_receipt(
                Path(tmp),
                status="ok", split="train", synthetic=True,
                formal_result=False, test_split_read=False,
                uses_external_weights=False, git_commit="abc1234",
                config={"a": 1},
            )
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse(data["formal_result"])
            self.assertFalse(data["test_split_read"])
            self.assertFalse(data["uses_external_weights"])


if __name__ == "__main__":
    unittest.main()