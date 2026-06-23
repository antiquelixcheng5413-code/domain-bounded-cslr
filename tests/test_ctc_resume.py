import tempfile
import unittest
from pathlib import Path

import numpy as np

from cslr.training.ctc import train_ctc_model


class CTCResumeTests(unittest.TestCase):
    def test_training_can_resume_from_latest_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = root / "features"
            features.mkdir()
            np.save(features / "train-1.npy", np.ones((48, 368), dtype=np.float32))
            np.save(features / "dev-1.npy", np.ones((48, 368), dtype=np.float32))
            manifest = root / "manifest.csv"
            manifest.write_text(
                "sample_id,video,label,signer,session,split\n"
                "train-1,train.mp4,A,s1,x,train\n"
                "dev-1,dev.mp4,A,s2,x,validation\n",
                encoding="utf-8",
            )
            vocabulary = root / "vocab.txt"
            vocabulary.write_text("A\n", encoding="utf-8")
            model_config = root / "model.yaml"
            model_config.write_text(
                "input_size: 368\nhidden_size: 4\nnum_layers: 1\ndropout: 0.0\n"
                "bidirectional: false\n",
                encoding="utf-8",
            )
            first_config = root / "first.yaml"
            first_config.write_text(
                "seed: 42\nbatch_size: 1\nepochs: 1\nlearning_rate: 0.001\n"
                "weight_decay: 0.0\nearly_stopping_patience: 2\ngradient_clip_norm: 1.0\n",
                encoding="utf-8",
            )
            second_config = root / "second.yaml"
            second_config.write_text(
                "seed: 42\nbatch_size: 1\nepochs: 2\nlearning_rate: 0.001\n"
                "weight_decay: 0.0\nearly_stopping_patience: 2\ngradient_clip_norm: 1.0\n",
                encoding="utf-8",
            )
            output = root / "ctc_v2.pt"

            first = train_ctc_model(
                manifest,
                features,
                vocabulary,
                model_config,
                first_config,
                output,
            )
            second = train_ctc_model(
                manifest,
                features,
                vocabulary,
                model_config,
                second_config,
                output,
                resume_path=Path(first["last_checkpoint"]),
            )

        self.assertTrue(Path(first["checkpoint"]).name.endswith("ctc_v2.pt"))
        self.assertEqual(Path(second["last_checkpoint"]).name, "ctc_v2.last.pt")
