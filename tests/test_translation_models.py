"""Unit tests for the assembled Part 3 model: training connectivity + forward."""

from __future__ import annotations

import unittest

import torch

from cslr.translation.models import Part3SpaMoModel


class ModelTest(unittest.TestCase):
    def _vocab(self) -> dict[str, int]:
        return {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "高": 4, "考": 5}

    def _model(self) -> Part3SpaMoModel:
        return Part3SpaMoModel(
            modalities=("rgb", "motion", "landmark"),
            hidden_dim=16,
            fusion_heads=2,
            fusion_layers=1,
            fusion_ff=32,
            dropout=0.0,
            vocab=self._vocab(),
            gloss_vocab_size=None,
            max_target_len=16,
            decoder_layers=1,
            decoder_heads=2,
            device="cpu",
        )

    def test_forward_with_target_trainable(self) -> None:
        model = self._model()
        model.train()
        features = {
            "rgb": torch.randn(2, 4, 16),
            "motion": torch.randn(2, 3, 16),
            "landmark": torch.randn(2, 4, 368),
        }
        masks = {k: torch.ones(v.shape[:2], dtype=torch.bool) for k, v in features.items()}
        out = model(features, masks, target_texts=["高考", "高考"])
        self.assertIsNotNone(out.loss)
        self.assertIsNotNone(out.logits)
        loss = out.loss  # type: ignore[assignment]
        loss.backward()
        params = [p for p in model.parameters() if p.requires_grad]
        self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in params))

    def test_tiny_model_overfits_synthetic(self) -> None:
        """Acceptance P3-07: tiny model overfits a tiny synthetic dataset."""
        torch.manual_seed(0)
        model = self._model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        features = {
            "rgb": torch.randn(4, 4, 16),
            "motion": torch.randn(4, 3, 16),
            "landmark": torch.randn(4, 4, 368),
        }
        masks = {k: torch.ones(v.shape[:2], dtype=torch.bool) for k, v in features.items()}
        texts = ["高考", "高考", "高考考的", "考高考"]
        losses: list[float] = []
        for _ in range(25):
            optimizer.zero_grad()
            out = model(features, masks, target_texts=texts)
            assert out.loss is not None
            out.loss.backward()
            optimizer.step()
            losses.append(float(out.loss.item()))
        self.assertLess(losses[-1], losses[0])
        # should drop well below the random-init baseline
        self.assertLess(losses[-1], 0.5)

    def test_generation(self) -> None:
        model = self._model()
        model.eval()
        features = {"rgb": torch.randn(1, 4, 16)}
        masks = {"rgb": torch.ones(1, 4, dtype=torch.bool)}
        with torch.no_grad():
            out = model(features, masks, generate=True, max_gen_len=8)
        self.assertEqual(len(out.generated_texts or []), 1)


if __name__ == "__main__":
    unittest.main()