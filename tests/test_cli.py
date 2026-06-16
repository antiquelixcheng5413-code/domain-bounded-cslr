import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from cslr.cli import main
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
                "train-00001,A,2023年高考到了。,2/0/2/3/高/考/时间/到/。,\n",
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
        self.assertEqual(records[0].label, "2023年高考到了。")


if __name__ == "__main__":
    unittest.main()
