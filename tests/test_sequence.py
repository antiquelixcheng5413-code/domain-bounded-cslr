import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cslr.features.extractor import MediaPipeHolisticExtractor
from cslr.features.sequence import resample_indices, resample_sequence


class SequenceTests(unittest.TestCase):
    def test_preserves_endpoints(self) -> None:
        self.assertEqual(resample_indices(3, 5), [0, 0, 1, 2, 2])

    def test_resamples_values(self) -> None:
        result = resample_sequence([[0.0], [1.0], [2.0]], 2)
        self.assertEqual(result, [[0.0], [2.0]])

    def test_image_sequence_paths_are_sorted(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "00010.jpg").write_bytes(b"")
            (root / "notes.txt").write_text("skip", encoding="utf-8")
            (root / "00002.jpg").write_bytes(b"")
            (root / "00001.png").write_bytes(b"")

            paths = MediaPipeHolisticExtractor._image_paths(root)

        self.assertEqual([path.name for path in paths], ["00001.png", "00002.jpg", "00010.jpg"])


if __name__ == "__main__":
    unittest.main()
