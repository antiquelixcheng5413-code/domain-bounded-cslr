import unittest

from cslr.features.sequence import resample_indices, resample_sequence


class SequenceTests(unittest.TestCase):
    def test_preserves_endpoints(self) -> None:
        self.assertEqual(resample_indices(3, 5), [0, 0, 1, 2, 2])

    def test_resamples_values(self) -> None:
        result = resample_sequence([[0.0], [1.0], [2.0]], 2)
        self.assertEqual(result, [[0.0], [2.0]])


if __name__ == "__main__":
    unittest.main()
