import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cslr.inference.ctc import CTCOnnxRecognizer
from cslr.models.ctc import build_ctc_model
from cslr.training.ctc import export_ctc_checkpoint_to_onnx, load_ctc_checkpoint


class CTCExportTests(unittest.TestCase):
    def test_ctc_v2_checkpoint_and_onnx_load(self) -> None:
        config = {
            "input_size": 368,
            "hidden_size": 4,
            "num_layers": 1,
            "dropout": 0.0,
            "bidirectional": False,
        }
        model = build_ctc_model(config, class_count=3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "ctc_v2.pt"
            onnx_path = root / "ctc_v2.onnx"
            torch.save(
                {
                    "model_kind": "ctc_v2",
                    "model_config": config,
                    "tokens": ["A", "B"],
                    "blank_index": 0,
                    "seed": 42,
                    "feature_receipt_sha256": "test",
                    "best_validation_corpus_wer": 1.0,
                    "vocabulary_sha256": "test",
                    "state_dict": model.state_dict(),
                },
                checkpoint_path,
            )

            restored_model, vocabulary, _ = load_ctc_checkpoint(checkpoint_path)
            exported = export_ctc_checkpoint_to_onnx(checkpoint_path, onnx_path)
            recognizer = CTCOnnxRecognizer.v2(onnx_path)
            result = recognizer.predict(np.zeros((48, 368), dtype=np.float32))

        self.assertEqual(vocabulary.blank_index, 0)
        self.assertEqual(restored_model(torch.zeros(1, 48, 368)).shape[-1], 3)
        self.assertEqual(exported["model"], str(onnx_path))
        self.assertEqual(result["model_kind"], "ctc_v2")


if __name__ == "__main__":
    unittest.main()
