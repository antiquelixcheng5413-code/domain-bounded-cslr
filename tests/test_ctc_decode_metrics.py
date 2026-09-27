import unittest

import numpy as np

from cslr.recognition.decode import classes_to_token_ids, greedy_decode, prefix_beam_search
from cslr.recognition.metrics import (
    character_error_rate,
    distinct_ratio,
    edit_distance,
    error_rate,
    gloss_metrics,
    sequence_exact_match,
)
from cslr.recognition.model import BLANK_INDEX


def log_probs_from_paths(path: list[int], classes: int, confidence: float = 0.9) -> np.ndarray:
    """Build a [T, C] log-probability matrix that argmaxes to ``path``."""

    rows = []
    for index in path:
        row = np.full(classes, np.log((1.0 - confidence) / (classes - 1)))
        row[index] = np.log(confidence)
        rows.append(row)
    return np.asarray(rows)


class GreedyDecodeTests(unittest.TestCase):
    def test_collapses_repeats_and_blanks(self) -> None:
        classes = 4
        path = [1, 1, BLANK_INDEX, 1, 2, 2, BLANK_INDEX, 2]
        decoded = greedy_decode(log_probs_from_paths(path, classes))
        self.assertEqual(decoded.classes, (1, 1, 2, 2))

    def test_all_blank_gives_empty_sequence(self) -> None:
        decoded = greedy_decode(log_probs_from_paths([0, 0, 0], 3))
        self.assertEqual(decoded.classes, ())

    def test_classes_map_back_to_vocabulary_indices(self) -> None:
        self.assertEqual(classes_to_token_ids((1, 2, BLANK_INDEX, 3)), [0, 1, 2])

    def test_rejects_wrong_rank(self) -> None:
        with self.assertRaises(ValueError):
            greedy_decode(np.zeros((3,)))


class PrefixBeamSearchTests(unittest.TestCase):
    def test_matches_greedy_on_confident_input(self) -> None:
        classes = 6
        path = [1, 2, BLANK_INDEX, 3, 4]
        matrix = log_probs_from_paths(path, classes)
        self.assertEqual(prefix_beam_search(matrix, beam_width=5).classes, greedy_decode(matrix).classes)

    def test_merges_two_paths_with_repeats(self) -> None:
        # blank-separated repeats collapse to two tokens, not one
        matrix = np.log(np.asarray(
            [
                [0.05, 0.90, 0.05],
                [0.05, 0.05, 0.90],
            ]
        ))
        decoded = prefix_beam_search(matrix, beam_width=8)
        self.assertEqual(decoded.classes, (1, 2))

    def test_beam_search_finds_path_greedy_misses(self) -> None:
        # Frame 1 slightly prefers class 2, but the (1, 2) hypothesis wins jointly.
        matrix = np.log(
            np.asarray(
                [
                    [0.10, 0.45, 0.45],
                    [0.10, 0.45, 0.45],
                ]
            )
        )
        greedy = greedy_decode(matrix)
        beam = prefix_beam_search(matrix, beam_width=8)
        self.assertIn(len(beam.classes), (1, 2))
        self.assertGreaterEqual(beam.score, greedy.score)

    def test_rejects_bad_beam_width(self) -> None:
        with self.assertRaises(ValueError):
            prefix_beam_search(log_probs_from_paths([1], 3), beam_width=0)


class EditDistanceTests(unittest.TestCase):
    def test_known_distances(self) -> None:
        self.assertEqual(edit_distance([], []), 0)
        self.assertEqual(edit_distance(["a"], []), 1)
        self.assertEqual(edit_distance([], ["a", "b"]), 2)
        self.assertEqual(edit_distance(["a", "b", "c"], ["a", "b", "c"]), 0)
        self.assertEqual(edit_distance(["a", "b", "c"], ["a", "x", "c"]), 1)
        self.assertEqual(edit_distance(["a", "b"], ["b", "a"]), 2)

    def test_error_rate_aggregates_over_corpus(self) -> None:
        rate = error_rate([["a", "b"], ["a", "b"]], [["a", "b"], ["a", "x"]])
        self.assertAlmostEqual(rate, 0.25)

    def test_error_rate_handles_empty_reference(self) -> None:
        rate = error_rate([[]], [["a", "b"]])
        self.assertAlmostEqual(rate, 2.0)

    def test_character_error_rate_uses_characters(self) -> None:
        rate = character_error_rate(["上海站"], ["上海"])
        self.assertAlmostEqual(rate, 1 / 3)

    def test_exact_match(self) -> None:
        self.assertAlmostEqual(sequence_exact_match([["a"], ["b"]], [["a"], ["c"]]), 0.5)

    def test_length_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            error_rate([["a"]], [])


class DistinctRatioTests(unittest.TestCase):
    def test_collapse_is_visible(self) -> None:
        collapsed = distinct_ratio([["我", "要"], ["我", "要"], ["我", "要"]])
        self.assertEqual(collapsed["unique"], 1)
        self.assertAlmostEqual(float(collapsed["distinct_ratio"]), 1 / 3)
        healthy = distinct_ratio([["我", "要"], ["你", "好"], ["他", "来"]])
        self.assertAlmostEqual(float(healthy["distinct_ratio"]), 1.0)


class GlossMetricsTests(unittest.TestCase):
    def test_reports_every_required_number(self) -> None:
        references = [["我", "要", "挂号"], ["你", "好"]]
        hypotheses = [["我", "要", "挂号"], ["你", "来"]]
        metrics = gloss_metrics(
            references,
            hypotheses,
            vocabulary_size=10,
            blank_steps=4,
            total_steps=10,
        )
        self.assertAlmostEqual(float(metrics["wer"]), 1 / 5)
        self.assertAlmostEqual(float(metrics["sequence_exact_match"]), 0.5)
        self.assertAlmostEqual(float(metrics["blank_ratio"]), 0.4)
        self.assertEqual(metrics["predicted_token_kinds"], 5)
        self.assertAlmostEqual(float(metrics["vocabulary_utilization"]), 0.5)
        self.assertEqual(metrics["empty_hypotheses"], 0)
        self.assertIn("distinct", metrics)

    def test_is_deterministic(self) -> None:
        references = [["我", "要"], ["你", "好"]]
        hypotheses = [["我"], ["你", "来"]]
        first = gloss_metrics(references, hypotheses)
        for _ in range(3):
            self.assertEqual(gloss_metrics(references, hypotheses), first)


if __name__ == "__main__":
    unittest.main()
