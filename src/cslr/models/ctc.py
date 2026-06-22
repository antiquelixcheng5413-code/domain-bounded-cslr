from __future__ import annotations

from typing import Any

import torch
from torch import nn


class CTCModel(nn.Module):
    """Frame-wise CTC logits over a CE-CSL Gloss vocabulary plus one blank class."""

    def __init__(
        self,
        input_size: int,
        class_count: int,
        hidden_size: int = 512,
        num_layers: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        output_size = hidden_size * (2 if bidirectional else 1)
        self.projection = nn.Linear(output_size, class_count)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.lstm(features)
        return self.projection(encoded)


class LegacyCTCModel(nn.Module):
    """Exact parameter structure of the delivered teammate CTC checkpoint."""

    def __init__(self, vocabulary_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=368,
            hidden_size=512,
            num_layers=3,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )
        self.fc = nn.Linear(1024, vocabulary_size + 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.lstm(features)
        return self.fc(encoded)


def build_ctc_model(config: dict[str, Any], class_count: int) -> CTCModel:
    return CTCModel(
        input_size=int(config["input_size"]),
        class_count=class_count,
        hidden_size=int(config.get("hidden_size", 512)),
        num_layers=int(config.get("num_layers", 3)),
        dropout=float(config.get("dropout", 0.3)),
        bidirectional=bool(config.get("bidirectional", True)),
    )


def build_legacy_ctc_model(vocabulary_size: int) -> LegacyCTCModel:
    return LegacyCTCModel(vocabulary_size=vocabulary_size)
