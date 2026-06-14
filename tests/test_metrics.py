import unittest

from cslr.evaluation.metrics import classification_metrics


class MetricTests(unittest.TestCase):
    def test_perfect_predictions(self) -> None:
        result = classification_metrics(["a", "b", "a"], ["a", "b", "a"])
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["macro_f1"], 1.0)

    def test_length_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            classification_metrics(["a"], [])


if __name__ == "__main__":
    unittest.main()
