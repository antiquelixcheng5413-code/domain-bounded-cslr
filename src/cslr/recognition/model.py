"""CTC model for ordered gloss recognition over cached feature sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

BLANK_INDEX = 0


@dataclass(frozen=True)
class CTCConfig:
    """Temporal model configuration. Vocabulary size excludes the CTC blank."""

    input_size: int
    vocabulary_size: int
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True
    projection_size: int = 256
    subsample_stride: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_size": self.input_size,
            "vocabulary_size": self.vocabulary_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "bidirectional": self.bidirectional,
            "projection_size": self.projection_size,
            "subsample_stride": self.subsample_stride,
            "blank_index": BLANK_INDEX,
        }


class CTCRecognizer(nn.Module):
    """LayerNorm -> projection -> (optional stride) -> BiLSTM -> CTC logits.

    The blank token is class 0, so the classifier emits ``vocabulary_size + 1``
    logits and a vocabulary index ``i`` corresponds to class ``i + 1``.
    """

    def __init__(self, config: CTCConfig) -> None:
        super().__init__()
        if config.vocabulary_size < 1:
            raise ValueError("vocabulary_size must be at least 1")
        if config.subsample_stride < 1:
            raise ValueError("subsample_stride must be at least 1")
        self.config = config
        self.normalize = nn.LayerNorm(config.input_size)
        self.projection = nn.Sequential(
            nn.Linear(config.input_size, config.projection_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.subsample = (
            nn.Identity()
            if config.subsample_stride == 1
            else nn.Conv1d(
                config.projection_size,
                config.projection_size,
                kernel_size=config.subsample_stride * 2,
                stride=config.subsample_stride,
                padding=config.subsample_stride // 2,
            )
        )
        self.temporal = nn.LSTM(
            input_size=config.projection_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            # torch warns and applies no dropout for a single layer; keep it explicit so the
            # forward pass is deterministic when num_layers == 1.
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=config.bidirectional,
            batch_first=True,
        )
        output_size = config.hidden_size * (2 if config.bidirectional else 1)
        self.classifier = nn.Linear(output_size, config.vocabulary_size + 1)

    @property
    def num_classes(self) -> int:
        return self.config.vocabulary_size + 1

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Sequence lengths after the optional stride subsampling."""

        stride = self.config.subsample_stride
        if stride == 1:
            return input_lengths
        lengths = torch.div(input_lengths, stride, rounding_mode="floor")
        return torch.clamp(lengths, min=1)

    def forward(self, features: torch.Tensor, input_lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Return logits shaped ``[B, T', V + 1]``."""

        if features.dim() != 3:
            raise ValueError(f"expected [B, T, D] features, got shape {tuple(features.shape)}")
        if features.shape[2] != self.config.input_size:
            raise ValueError(
                f"expected input_size {self.config.input_size}, got {features.shape[2]}"
            )
        if input_lengths is not None:
            # Zero the padded feature rows: the CTC loss and decoding only ever read the first
            # ``output_lengths`` steps, and this keeps garbage in the padded tail from reaching
            # the recurrence at all.
            features = self._mask_padding(features, input_lengths)
        hidden = self.projection(self.normalize(features))
        if self.config.subsample_stride != 1:
            hidden = self.subsample(hidden.transpose(1, 2)).transpose(1, 2)
        encoded, _ = self.temporal(hidden)
        logits = self.classifier(encoded)
        return logits

    @staticmethod
    def _mask_padding(features: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        """Zero every timestep at or beyond ``input_lengths``."""

        steps = features.shape[1]
        positions = torch.arange(steps, device=features.device).unsqueeze(0)
        mask = positions < input_lengths.unsqueeze(1)
        return features * mask.unsqueeze(-1).to(features.dtype)

    @torch.no_grad()
    def log_probs(self, features: torch.Tensor, input_lengths: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.forward(features, input_lengths)
        return torch.log_softmax(logits.float(), dim=-1)


def build_ctc_model(config: CTCConfig) -> CTCRecognizer:
    return CTCRecognizer(config)


def ctc_config_from_dict(payload: dict[str, Any]) -> CTCConfig:
    """Rebuild a ``CTCConfig`` from its serialised form.

    ``CTCConfig.as_dict()`` records ``blank_index`` for inspection, which is not a constructor
    argument, so unknown keys are filtered rather than passed through.
    """

    import dataclasses

    fields = {field.name for field in dataclasses.fields(CTCConfig)}
    known = {key: value for key, value in payload.items() if key in fields}
    return CTCConfig(**known)
