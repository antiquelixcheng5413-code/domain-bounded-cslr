"""Unit tests for Part 3 encoder interfaces and tiny implementations."""

from __future__ import annotations

import unittest

import torch

from cslr.translation.encoders import (
    TinyLandmarkEncoder,
    TinyMotionEncoder,
    TinyRGBEncoder,
    build_encoder,
)


class EncodersTest(unittest.TestCase):
    def _device(self) -> torch.device:
        return torch.device("cpu")

    def test_rgb_encoder(self) -> None:
        enc = TinyRGBEncoder(input_dim=32, hidden_dim=16)
        x = torch.randn(2, 4, 32)
        tokens, mask = enc.encode(x)
        self.assertEqual(tokens.shape, (2, 4, 16))
        self.assertEqual(mask.shape, (2, 4))

    def test_motion_encoder_preserves_length(self) -> None:
        # Adjacent-frame differencing lives in feature extraction (Phase 2), so
        # the encoder only projects tokens and must not shorten the sequence.
        enc = TinyMotionEncoder(input_dim=32, hidden_dim=16)
        x = torch.randn(2, 4, 32)
        tokens, mask = enc.encode(x)
        self.assertEqual(tokens.shape, (2, 4, 16))  # length preserved

    def test_landmark_encoder(self) -> None:
        enc = TinyLandmarkEncoder(landmark_dim=368, hidden_dim=16)
        x = torch.randn(2, 4, 368)
        tokens, mask = enc.encode(x)
        self.assertEqual(tokens.shape, (2, 4, 16))

    def test_build_encoder_routes(self) -> None:
        self.assertIsInstance(build_encoder("rgb", input_dim=32, hidden_dim=16), TinyRGBEncoder)
        self.assertIsInstance(build_encoder("motion", input_dim=32, hidden_dim=16), TinyMotionEncoder)
        self.assertIsInstance(build_encoder("landmark", input_dim=368, hidden_dim=16), TinyLandmarkEncoder)

    def test_unknown_modality(self) -> None:
        with self.assertRaises(ValueError):
            build_encoder("audio", input_dim=16, hidden_dim=16)


if __name__ == "__main__":
    unittest.main()