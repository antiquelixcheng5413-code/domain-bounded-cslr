import unittest
from pathlib import Path

from cslr.contracts import SampleRecord


class SampleRecordTests(unittest.TestCase):
    def test_accepts_relative_video_path(self) -> None:
        record = SampleRecord("s1", Path("clips/one.mp4"), "REGISTER", "p01", "day1", "train")
        record.validate()

    def test_rejects_path_escape(self) -> None:
        record = SampleRecord("s1", Path("../secret.mp4"), "REGISTER", "p01", "day1", "train")
        with self.assertRaises(ValueError):
            record.validate()

    def test_rejects_unknown_split(self) -> None:
        record = SampleRecord("s1", Path("one.mp4"), "REGISTER", "p01", "day1", "random")
        with self.assertRaises(ValueError):
            record.validate()


if __name__ == "__main__":
    unittest.main()
