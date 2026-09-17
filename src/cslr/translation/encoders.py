"""RGB / Motion / Landmark encoder interfaces and lightweight Tiny implementations.

Any real pretrained encoder (CLIP / DINOv2 / VideoMAE) must be wrapped behind an
adapter so training and evaluation never depend on a private output format.
"""

from __future__ import annotations

import torch
from torch import nn


class BaseSequenceEncoder(nn.Module):
    """Base contract: input -> (tokens, mask). Output dim follows hidden_dim."""

    def encode(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def forward(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode(features, mask)


class TinyRGBEncoder(BaseSequenceEncoder):
    """Dimension projection for the smoke test; keeps the abstract interface."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def encode(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.proj(features)
        if mask is None:
            mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        return tokens, mask


class TinyMotionEncoder(BaseSequenceEncoder):
    """Projects motion features (frame differences) to the shared space.

    The T-1 adjacent-frame differencing is performed by the feature-extraction
    stage (Phase 2), not here: the encoder only projects whatever motion tokens
    it receives, preserving their length.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def encode(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.proj(features)
        if mask is None:
            mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        return tokens, mask


class TinyLandmarkEncoder(BaseSequenceEncoder):
    """Project 368-dim MediaPipe landmarks into the shared fusion space."""

    def __init__(self, landmark_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.landmark_dim = landmark_dim
        self.hidden_dim = hidden_dim
        self.proj = nn.Sequential(
            nn.LayerNorm(landmark_dim),
            nn.Linear(landmark_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def encode(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.proj(features)
        if mask is None:
            mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        return tokens, mask


def build_encoder(modality: str, *, input_dim: int, hidden_dim: int, dropout: float = 0.0) -> BaseSequenceEncoder:
    if modality == "rgb":
        return TinyRGBEncoder(input_dim, hidden_dim, dropout)
    if modality == "motion":
        return TinyMotionEncoder(input_dim, hidden_dim, dropout)
    if modality == "landmark":
        return TinyLandmarkEncoder(input_dim if modality == "landmark" else 368, hidden_dim, dropout)
    raise ValueError(f"unknown modality {modality!r}")