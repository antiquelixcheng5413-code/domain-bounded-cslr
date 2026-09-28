import tempfile
import unittest
from pathlib import Path

import torch

from cslr.recognition.contrastive_pretrain import (
    ContrastiveEncoder,
    GlossEmbedding,
    contrastive_loss,
    copy_frontend_weights,
    mean_pool_hidden,
    positive_gloss_ids,
)
from cslr.recognition.model import CTCConfig, CTCRecognizer


class MeanPoolTests(unittest.TestCase):
    def test_padding_never_contributes(self) -> None:
        hidden = torch.tensor(
            [[[1.0, 1.0], [3.0, 3.0], [0.0, 0.0], [0.0, 0.0]]], dtype=torch.float32
        )
        pooled = mean_pool_hidden(hidden, torch.tensor([2], dtype=torch.long))
        self.assertTrue(torch.allclose(pooled, torch.tensor([[2.0, 2.0]])))

    def test_single_frame_is_that_frame(self) -> None:
        hidden = torch.tensor([[[5.0, -1.0], [7.0, 2.0]]], dtype=torch.float32)
        pooled = mean_pool_hidden(hidden, torch.tensor([1], dtype=torch.long))
        self.assertTrue(torch.allclose(pooled, torch.tensor([[5.0, -1.0]])))


class PositiveGlossIdsTests(unittest.TestCase):
    def test_excludes_unknown_index_zero(self) -> None:
        from cslr.recognition.dataset import SequenceSample

        samples = [
            SequenceSample(sample_id="a", features=None, tokens=(), token_ids=(0, 3, 3, 1)),
            SequenceSample(sample_id="b", features=None, tokens=(), token_ids=(0,)),
        ]
        self.assertEqual(positive_gloss_ids(samples), [[1, 3], []])

    def test_deduplicates_repeated_tokens(self) -> None:
        from cslr.recognition.dataset import SequenceSample

        samples = [
            SequenceSample(sample_id="a", features=None, tokens=(), token_ids=(2, 2, 2)),
        ]
        self.assertEqual(positive_gloss_ids(samples), [[2]])


class ContrastiveLossTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.embeddings = torch.nn.functional.normalize(
            torch.randn(6, 4), dim=-1
        )  # vocab of 6

    def test_aligned_anchor_scores_lower_loss(self) -> None:
        positives = [[1]]
        aligned = self.embeddings[1:2].clone()  # anchor equal to the positive token
        misaligned = self.embeddings[2:3].clone()  # anchor equal to a negative token
        loss_aligned = contrastive_loss(aligned, positives, self.embeddings, 0.07)
        loss_misaligned = contrastive_loss(misaligned, positives, self.embeddings, 0.07)
        self.assertLess(loss_aligned.item(), loss_misaligned.item())

    def test_multiple_positives_average(self) -> None:
        positives = [[1, 2]]
        anchor = self.embeddings[1:2].clone()
        loss = contrastive_loss(anchor, positives, self.embeddings, 0.07)
        self.assertTrue(torch.isfinite(loss))

    def test_empty_positives_returns_zero(self) -> None:
        loss = contrastive_loss(self.embeddings[:2], [[], []], self.embeddings, 0.07)
        self.assertEqual(loss.item(), 0.0)

    def test_loss_is_differentiable(self) -> None:
        positives = [[1, 2]]
        anchor = self.embeddings[1:2].clone().requires_grad_(True)
        loss = contrastive_loss(anchor, positives, self.embeddings, 0.07)
        loss.backward()
        self.assertIsNotNone(anchor.grad)


class FrontendCopyTests(unittest.TestCase):
    def test_copies_normalize_and_projection_into_ctc(self) -> None:
        encoder = ContrastiveEncoder(input_size=8, projection_size=16)
        recognizer = CTCRecognizer(
            CTCConfig(input_size=8, vocabulary_size=5, hidden_size=4, projection_size=16)
        )
        encoder_state = {key: value.clone() for key, value in encoder.state_dict().items()}
        missing, unexpected, skipped = copy_frontend_weights(encoder_state, recognizer)
        # every recognizer parameter except LSTM/classifier must be missing from the copy
        self.assertIn("temporal.weight_ih_l0", missing)
        self.assertIn("classifier.weight", missing)
        self.assertEqual(unexpected, [])
        self.assertEqual(skipped, [])
        for key in ("normalize.weight", "normalize.bias", "projection.0.weight", "projection.0.bias"):
            self.assertTrue(
                torch.equal(recognizer.state_dict()[key], encoder_state[key]),
                f"{key} was not copied",
            )

    def test_non_matching_encoder_width_is_skipped(self) -> None:
        encoder = ContrastiveEncoder(input_size=8, projection_size=16)
        recognizer = CTCRecognizer(
            CTCConfig(input_size=8, vocabulary_size=5, hidden_size=4, projection_size=32)
        )
        missing, unexpected, skipped = copy_frontend_weights(encoder.state_dict(), recognizer)
        self.assertIn("projection.0.weight", skipped)
        self.assertIn("projection.0.bias", skipped)
        self.assertEqual(unexpected, [])


class GlossEmbeddingTests(unittest.TestCase):
    def test_embeddings_are_unit_normed(self) -> None:
        embedding = GlossEmbedding(10, 8)
        table = embedding.normalized_embeddings()
        self.assertLess((table.norm(dim=-1) - 1.0).abs().max().item(), 1e-5)


class PretrainCheckpointRoundTripTests(unittest.TestCase):
    def test_checkpoint_saves_vocabulary_and_frontend(self) -> None:
        from cslr.recognition.contrastive_pretrain import ContrastiveConfig

        config = ContrastiveConfig(input_size=8, projection_size=16)
        self.assertEqual(config.as_dict()["temperature"], 0.07)

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "pretrain.pt"
            encoder = ContrastiveEncoder(8, 16)
            torch.save({"state_dict": encoder.state_dict()}, checkpoint)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertIn("normalize.weight", payload["state_dict"])


if __name__ == "__main__":
    unittest.main()
