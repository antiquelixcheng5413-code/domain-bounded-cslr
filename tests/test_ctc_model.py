import unittest

import numpy as np
import torch

from cslr.recognition.model import BLANK_INDEX, CTCConfig, CTCRecognizer
from cslr.recognition.training import ctc_loss, pad_targets


def tiny_config(**overrides) -> CTCConfig:
    base = dict(
        input_size=8,
        vocabulary_size=5,
        hidden_size=16,
        num_layers=1,
        dropout=0.0,
        projection_size=12,
        subsample_stride=1,
    )
    base.update(overrides)
    return CTCConfig(**base)  # type: ignore[arg-type]


class CTCModelTests(unittest.TestCase):
    def test_logits_shape_uses_vocabulary_plus_blank(self) -> None:
        model = CTCRecognizer(tiny_config())
        features = torch.randn(3, 20, 8)
        logits = model(features)
        self.assertEqual(tuple(logits.shape), (3, 20, 6))
        self.assertEqual(model.num_classes, 6)

    def test_blank_index_is_zero(self) -> None:
        self.assertEqual(BLANK_INDEX, 0)

    def test_subsample_stride_shortens_the_sequence(self) -> None:
        model = CTCRecognizer(tiny_config(subsample_stride=2))
        features = torch.randn(2, 40, 8)
        lengths = torch.tensor([40, 30])
        logits = model(features, lengths)
        self.assertLess(logits.shape[1], 40)
        output_lengths = model.output_lengths(lengths)
        self.assertTrue(torch.all(output_lengths <= logits.shape[1]))

    def test_output_lengths_never_drop_below_one(self) -> None:
        model = CTCRecognizer(tiny_config(subsample_stride=2))
        output_lengths = model.output_lengths(torch.tensor([1]))
        self.assertTrue(torch.all(output_lengths >= 1))

    def test_rejects_mismatched_input_size(self) -> None:
        model = CTCRecognizer(tiny_config(input_size=8))
        with self.assertRaises(ValueError):
            model(torch.randn(1, 5, 7))

    def test_rejects_non_three_dimensional_input(self) -> None:
        model = CTCRecognizer(tiny_config())
        with self.assertRaises(ValueError):
            model(torch.randn(5, 8))

    def test_garbage_padding_never_reaches_valid_outputs(self) -> None:
        """The invariant CTC actually depends on: padded content must be ignored exactly."""

        torch.manual_seed(0)
        model = CTCRecognizer(tiny_config())
        model.eval()
        features = torch.randn(1, 10, 8)
        garbage = features.clone()
        garbage[0, 6:] = torch.randn(4, 8) * 5.0
        lengths = torch.tensor([6])
        with torch.no_grad():
            clean = model(features, lengths)[0, :6]
            dirty = model(garbage, lengths)[0, :6]
        self.assertTrue(torch.equal(clean, dirty))

    def test_prefix_is_stable_across_total_length(self) -> None:
        # torch's CPU LSTM is not bit-identical for the same prefix at different total lengths
        # (measured ~0.03 in logit space), so this asserts closeness rather than equality.
        torch.manual_seed(0)
        model = CTCRecognizer(tiny_config())
        model.eval()
        features = torch.randn(1, 10, 8)
        with torch.no_grad():
            short = model(features, torch.tensor([6]))[0, :6]
            long = model(features, torch.tensor([10]))[0, :6]
        self.assertTrue(torch.allclose(short, long, atol=0.05))

    def test_ctc_loss_decreases_after_one_step(self) -> None:
        torch.manual_seed(0)
        model = CTCRecognizer(tiny_config())
        features = torch.randn(4, 25, 8)
        padded = pad_targets([1, 2, 3, 4], [1, 1, 1, 1])
        target_lengths = torch.tensor([1, 1, 1, 1])
        input_lengths = torch.tensor([25, 25, 25, 25])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)

        def compute_loss() -> float:
            logits = model(features, input_lengths)
            output_lengths = model.output_lengths(input_lengths)
            return float(
                ctc_loss(logits, input_lengths, padded, target_lengths, output_lengths).item()
            )

        before = compute_loss()
        for _ in range(5):
            optimizer.zero_grad()
            logits = model(features, input_lengths)
            output_lengths = model.output_lengths(input_lengths)
            loss = ctc_loss(logits, input_lengths, padded, target_lengths, output_lengths)
            loss.backward()
            optimizer.step()
        after = compute_loss()
        self.assertLess(after, before)

    def test_pad_targets_builds_two_dimensional_tensor(self) -> None:
        padded = pad_targets([1, 2, 3], [2, 1])
        self.assertEqual(tuple(padded.shape), (2, 2))
        self.assertEqual(padded[0].tolist(), [1, 2])
        self.assertEqual(padded[1].tolist(), [3, BLANK_INDEX])

    def test_pad_targets_rejects_inconsistent_lengths(self) -> None:
        with self.assertRaises(ValueError):
            pad_targets([1, 2, 3], [1, 1])

    def test_log_probs_are_normalized(self) -> None:
        model = CTCRecognizer(tiny_config())
        log_probs = model.log_probs(torch.randn(2, 7, 8))
        total = log_probs.exp().sum(dim=-1)
        self.assertTrue(torch.allclose(total, torch.ones_like(total), atol=1e-4))

    def test_config_serialization_mentions_blank(self) -> None:
        payload = tiny_config().as_dict()
        self.assertEqual(payload["blank_index"], BLANK_INDEX)
        self.assertEqual(payload["vocabulary_size"], 5)


if __name__ == "__main__":
    unittest.main()
