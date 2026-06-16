import tempfile
import unittest
from pathlib import Path

import numpy as np

from cslr.contracts import QualityReport, SampleRecord
from cslr.features.batch import extract_manifest_features, write_extraction_report
from cslr.features.extractor import ExtractionResult


class FakeExtractor:
    def extract(self, source: Path) -> ExtractionResult:
        if not source.exists():
            raise FileNotFoundError(source)
        return ExtractionResult(
            features=np.ones((2, 3), dtype=np.float32),
            quality=QualityReport(
                total_frames=2,
                valid_frames=2,
                valid_ratio=1.0,
                accepted=True,
                warnings=[],
            ),
            source_frames=2,
        )


class BatchExtractionTests(unittest.TestCase):
    def test_extracts_manifest_records_to_feature_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            output_root = root / "features"
            video = data_root / "video" / "train" / "A" / "train-00001.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"video")
            records = [
                SampleRecord(
                    sample_id="train-00001",
                    video=Path("video/train/A/train-00001.mp4"),
                    label="2023年高考到了。",
                    signer="A",
                    session="train",
                    split="train",
                )
            ]

            summary = extract_manifest_features(
                records=records,
                data_root=data_root,
                output_root=output_root,
                extractor=FakeExtractor(),
            )

            saved = np.load(output_root / "train-00001.npy")

        self.assertEqual(summary.extracted, 1)
        self.assertEqual(summary.accepted, 1)
        self.assertEqual(saved.shape, (2, 3))

    def test_skips_existing_outputs_and_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            output_root = root / "features"
            output_root.mkdir()
            np.save(output_root / "train-00001.npy", np.zeros((1, 1), dtype=np.float32))
            report = root / "quality.csv"
            records = [
                SampleRecord(
                    sample_id="train-00001",
                    video=Path("video/train/A/train-00001.mp4"),
                    label="label",
                    signer="A",
                    session="train",
                    split="train",
                )
            ]

            summary = extract_manifest_features(
                records=records,
                data_root=data_root,
                output_root=output_root,
                extractor=FakeExtractor(),
            )
            count = write_extraction_report(report, summary.items)

            report_text = report.read_text(encoding="utf-8")

        self.assertEqual(summary.skipped, 1)
        self.assertEqual(count, 1)
        self.assertIn("existing output", report_text)


if __name__ == "__main__":
    unittest.main()
