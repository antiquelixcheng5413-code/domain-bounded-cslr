import unittest

import numpy as np

from cslr.evaluation.ctc import corpus_wer, ctc_greedy_decode, token_edit_counts


class CTCMetricTests(unittest.TestCase):
    def test_decoder_allows_repeated_token_after_blank(self) -> None:
        logits = np.asarray(
            [
                [0.0, 5.0, 0.0],
                [0.0, 4.0, 0.0],
                [6.0, 0.0, 0.0],
                [0.0, 5.0, 0.0],
            ],
            dtype=np.float32,
        )
        decoded = ctc_greedy_decode(logits, {1: "A", 2: "B"}, blank_index=0)
        self.assertEqual(decoded.tokens, ["A", "A"])
        self.assertGreater(decoded.path_score, 0.0)

    def test_corpus_wer_accumulates_substitution_deletion_and_insertion(self) -> None:
        first = token_edit_counts(["A", "B", "C"], ["A", "X", "C", "D"])
        second = token_edit_counts(["E", "F"], ["E"])
        result = corpus_wer([first, second])

        self.assertEqual(first.substitutions, 1)
        self.assertEqual(first.insertions, 1)
        self.assertEqual(second.deletions, 1)
        self.assertEqual(result["errors"], 3)
        self.assertEqual(result["reference_tokens"], 5)
        self.assertAlmostEqual(float(result["wer"]), 0.6)
