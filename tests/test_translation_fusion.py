"""Unit tests for the SpaMo-style fusion module (tri/dual/single modality)."""

from __future__ import annotations

import unittest

import torch

from cslr.translation.fusion import LightSpaMoFusion


class FusionTest(unittest.TestCase):
    def test_tri_modal_token_count(self) -> None:
        fusion = LightSpaMoFusion(
            modalities=("rgb", "motion", "landmark"),
            hidden_dim=16, num_heads=4, num_layers=1, feedforward_dim=32, dropout=0.0,
        )
        features = {"rgb": torch.randn(2, 4, 16), "motion": torch.randn(2, 3, 16), "landmark": torch.randn(2, 4, 368)}
        masks = {k: torch.ones(v.shape[:2], dtype=torch.bool) for k, v in features.items()}
        tokens, mask, count = fusion(features, masks)
        self.assertEqual(tokens.shape[1], 11)  # 4+3+4
        self.assertEqual(int(mask.sum().item()), 22)  # 2 samples × 11 tokens
        self.assertEqual(count, 11)  # per-sample actual token count

    def test_dual_modal(self) -> None:
        fusion = LightSpaMoFusion(
            modalities=("rgb", "landmark"),
            hidden_dim=16, num_heads=4, num_layers=1, feedforward_dim=32, dropout=0.0,
        )
        features = {"rgb": torch.randn(2, 4, 16), "landmark": torch.randn(2, 4, 368)}
        masks = {k: torch.ones(v.shape[:2], dtype=torch.bool) for k, v in features.items()}
        tokens, mask, count = fusion(features, masks)
        self.assertEqual(tokens.shape[1], 8)
        self.assertEqual(count, 8)

    def test_single_modal(self) -> None:
        fusion = LightSpaMoFusion(
            modalities=("rgb",),
            hidden_dim=16, num_heads=4, num_layers=1, feedforward_dim=32, dropout=0.0,
        )
        features = {"rgb": torch.randn(2, 4, 16)}

        def masks_for(d: dict) -> dict:
            return {k: torch.ones(v.shape[:2], dtype=torch.bool) for k, v in d.items()}

        tokens, mask, count = fusion(features, masks_for(features))
        self.assertEqual(tokens.shape[1], 4)
        self.assertEqual(count, 4)

    def test_no_modality_error(self) -> None:
        fusion = LightSpaMoFusion(
            modalities=("rgb",),
            hidden_dim=16, num_heads=4, num_layers=1, feedforward_dim=32, dropout=0.0,
        )
        with self.assertRaises(ValueError):
            fusion({})

    def test_padding_not_counted(self) -> None:
        fusion = LightSpaMoFusion(
            modalities=("rgb",),
            hidden_dim=16, num_heads=4, num_layers=1, feedforward_dim=32, dropout=0.0,
        )
        features = {"rgb": torch.randn(1, 4, 16)}
        mask = torch.zeros(1, 4, dtype=torch.bool)
        mask[0, :2] = True
        tokens, out_mask, count = fusion(features, {"rgb": mask})
        self.assertEqual(count, 2)


if __name__ == "__main__":
    unittest.main()