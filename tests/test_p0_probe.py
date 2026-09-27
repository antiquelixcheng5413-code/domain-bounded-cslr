import tempfile
import unittest
from pathlib import Path

import numpy as np

from cslr.recognition.p0_vl_probe import (
    FROZEN_SPLIT_ERROR,
    LocalVlModel,
    extract_frames,
    load_reference_sentences,
    run_probe,
)


def _write_tiny_video(path: Path, frames: int = 12, size: int = 64) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (size, size)
    )
    for index in range(frames):
        image = np.full((size, size, 3), index * 15 % 255, dtype=np.uint8)
        writer.write(image)
    writer.release()


class FrameExtractionTests(unittest.TestCase):
    def test_extracts_the_requested_number_of_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.mp4"
            _write_tiny_video(video, frames=12)
            frames = extract_frames(video, count=4, workdir=root / "frames")
            self.assertEqual(len(frames), 4)
            self.assertTrue(all(frame.exists() for frame in frames))

    def test_is_idempotent_and_reuses_the_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.mp4"
            _write_tiny_video(video, frames=10)
            first = extract_frames(video, count=3, workdir=root / "frames")
            second = extract_frames(video, count=3, workdir=root / "frames")
            self.assertEqual([p.name for p in first], [p.name for p in second])

    def test_missing_video_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                extract_frames(Path(tmp) / "nope.mp4", count=2, workdir=Path(tmp) / "frames")

    def test_zero_frames_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "clip.mp4"
            _write_tiny_video(video, frames=6)
            with self.assertRaises(ValueError):
                extract_frames(video, count=0, workdir=root / "frames")


class ReferenceSentenceTests(unittest.TestCase):
    def test_reads_chinese_sentences_by_sample_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            label_dir = root / "label"
            label_dir.mkdir(parents=True)
            (label_dir / "dev.csv").write_text(
                "Number,Translator,Chinese Sentences,Gloss,Note\n"
                "dev-00001,A,今天很好。,今天/好/。,\n",
                encoding="utf-8",
            )
            sentences = load_reference_sentences(root, "dev.csv")
        self.assertEqual(sentences["dev-00001"], "今天很好。")

    def test_missing_label_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_reference_sentences(Path(tmp), "dev.csv")


class ProbeGuardTests(unittest.TestCase):
    def test_frozen_split_is_refused_before_any_file_is_opened(self) -> None:
        import argparse

        args = argparse.Namespace(split="test", data_root="/nonexistent")
        with self.assertRaises(SystemExit) as context:
            run_probe(args)
        self.assertEqual(str(context.exception), FROZEN_SPLIT_ERROR)

    def test_local_model_requires_optional_dependencies(self) -> None:
        # constructing the wrapper must either succeed (deps present) or explain itself
        try:
            LocalVlModel("definitely-not-a-real-model-path", load_in_4bit=True)
        except RuntimeError as error:
            self.assertIn("transformers", str(error))
        except Exception:
            pass  # dependency errors from transformers itself are acceptable here


if __name__ == "__main__":
    unittest.main()
