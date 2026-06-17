import unittest

from cslr.evaluation.metrics import classification_metrics, multilabel_metrics


class MetricTests(unittest.TestCase):
    def test_perfect_predictions(self) -> None:
        result = classification_metrics(["a", "b", "a"], ["a", "b", "a"])
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["macro_f1"], 1.0)

    def test_length_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            classification_metrics(["a"], [])

    def test_multilabel_metrics(self) -> None:
        result = multilabel_metrics(
            expected=[["a", "b"], ["b"]],
            predicted=[["a"], ["b", "c"]],
            labels=["a", "b", "c"],
        )
        self.assertAlmostEqual(result["micro_f1"], 4 / 6)
        self.assertEqual(result["subset_accuracy"], 0.0)
        self.assertEqual(result["per_label"]["a"]["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
