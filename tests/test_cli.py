import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

from cslr.cli import build_parser, main
from cslr.data.manifest import read_manifest


class CliTests(unittest.TestCase):
    def test_build_manifest_uses_configured_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "ce-csl"
            (data_root / "label").mkdir(parents=True)
            (data_root / "video" / "train" / "A").mkdir(parents=True)
            (data_root / "video" / "train" / "A" / "train-00001.mp4").write_bytes(b"video")
            (data_root / "label" / "train.csv").write_text(
                "Number,Translator,Chinese Sentences,Gloss,Note\n"
                "train-00001,A,Sentence one,A/B/C,\n",
                encoding="utf-8",
            )
            config = root / "ce_csl.yaml"
            output = root / "manifest.csv"
            config.write_text(
                "\n".join(
                    [
                        "name: ce_csl_test",
                        "adapter: ce_csl_video",
                        f"data_root: {data_root.as_posix()}",
                        "adapter_options:",
                        "  split_files:",
                        "    train: label/train.csv",
                        "  video_splits:",
                        "    train: train",
                        "  label_column: Gloss",
                    ]
                ),
                encoding="utf-8",
            )
            stdout = StringIO()

            with redirect_stdout(stdout):
                exit_code = main(["build-manifest", str(config), "--output", str(output)])

            result = json.loads(stdout.getvalue())
            records = read_manifest(output)

        self.assertEqual(exit_code, 0)
        self.assertEqual(result["records"], 1)
        self.assertEqual(records[0].video.as_posix(), "video/train/A/train-00001.mp4")
        self.assertEqual(records[0].label, "A/B/C")

    def test_build_gloss_vocab_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            output = root / "vocab.csv"
            manifest.write_text(
                "sample_id,video,label,signer,session,split\n"
                "s1,a.mp4,A/B/C,p01,s,train\n"
                "s2,b.mp4,A/B/D,p02,s,train\n"
                "s3,c.mp4,A/E,p03,s,validation\n",
                encoding="utf-8",
            )
            stdout = StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "build-gloss-vocab",
                        str(manifest),
                        "--output",
                        str(output),
                        "--min-frequency",
                        "2",
                    ]
                )

            result = json.loads(stdout.getvalue())
            content = output.read_text(encoding="utf-8")

        self.assertEqual(exit_code, 0)
        self.assertEqual(result["tokens"], 2)
        self.assertIn("A,2", content)
        self.assertIn("B,2", content)

    def test_verify_features_writes_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            features = root / "features"
            receipt = root / "receipt.json"
            features.mkdir()
            manifest.write_text(
                "sample_id,video,label,signer,session,split\n"
                "s1,a.mp4,A,p01,s,train\n",
                encoding="utf-8",
            )
            np.save(features / "s1.npy", np.zeros((48, 368), dtype=np.float32))
            stdout = StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "verify-features",
                        str(manifest),
                        "--features",
                        str(features),
                        "--sha256",
                        "--receipt",
                        str(receipt),
                    ]
                )

            result = json.loads(stdout.getvalue())
            receipt_exists = receipt.exists()

        self.assertEqual(exit_code, 0)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(receipt_exists)

    def test_ctc_commands_are_registered(self) -> None:
        parser = build_parser()
        commands = parser._subparsers._group_actions[0].choices
        self.assertIn("train-ctc", commands)
        self.assertIn("evaluate-ctc", commands)
        self.assertIn("export-ctc", commands)
        self.assertIn("gpu-preflight", commands)
        self.assertIn("benchmark-ctc", commands)


if __name__ == "__main__":
    unittest.main()
