import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cslr.recognition.clip_features import sample_frames
from cslr.recognition.qwen_vl_features import FrozenQwenVisionEncoder


def _write_video(path: Path, frames: int = 20, size: int = 64) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (size, size))
    for index in range(frames):
        writer.write(np.full((size, size, 3), (index * 11) % 255, dtype=np.uint8))
    writer.release()


class SampleFramesTests(unittest.TestCase):
    def test_samples_the_requested_count_as_rgb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "clip.mp4"
            _write_video(video, frames=20)
            frames = sample_frames(video, count=5, size=32)
            self.assertEqual(len(frames), 5)
            self.assertEqual(frames[0].shape[2], 3)
            self.assertTrue(all(frame.dtype == np.uint8 for frame in frames))

    def test_missing_video_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                sample_frames(Path(tmp) / "nope.mp4", count=2)

    def test_zero_frames_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "clip.mp4"
            _write_video(video, frames=8)
            with self.assertRaises(ValueError):
                sample_frames(video, count=0)


class PerFramePoolingTests(unittest.TestCase):
    """The pooling rule is pure tensor reshaping, so it can be tested without the model."""

    def test_pooling_averages_tokens_inside_each_frame(self) -> None:
        encoder = FrozenQwenVisionEncoder.__new__(FrozenQwenVisionEncoder)
        encoder.torch = torch

        class Output:
            def __init__(self, pooled: torch.Tensor) -> None:
                self.pooler_output = pooled

        tokens_per_frame = 4
        width = 3
        pooled = torch.arange(2 * tokens_per_frame * width, dtype=torch.float32).reshape(
            2 * tokens_per_frame, width
        )
        result = encoder._pool_per_frame(Output(pooled), 2)
        self.assertEqual(tuple(result.shape), (2, width))
        expected_first = pooled[:tokens_per_frame].mean(dim=0)
        expected_second = pooled[tokens_per_frame:].mean(dim=0)
        self.assertTrue(torch.allclose(result[0], expected_first))
        self.assertTrue(torch.allclose(result[1], expected_second))

    def test_non_divisible_token_count_is_rejected(self) -> None:
        encoder = FrozenQwenVisionEncoder.__new__(FrozenQwenVisionEncoder)

        class Output:
            pooler_output = torch.zeros(7, 4)

        with self.assertRaises(RuntimeError):
            encoder._pool_per_frame(Output(), 2)

    def test_missing_pooler_output_is_rejected(self) -> None:
        encoder = FrozenQwenVisionEncoder.__new__(FrozenQwenVisionEncoder)

        class Output:
            pooler_output = None

        with self.assertRaises(RuntimeError):
            encoder._pool_per_frame(Output(), 1)


if __name__ == "__main__":
    unittest.main()
