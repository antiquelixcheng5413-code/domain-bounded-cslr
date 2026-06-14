from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
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
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(output_size, num_classes))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(features)
        return self.classifier(output[:, -1, :])


class TemporalBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int, dropout: float):
        super().__init__()
        padding = kernel_size // 2
        self.network = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(output_channels, output_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(),
        )
        self.residual = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv1d(input_channels, output_channels, 1)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.network(features) + self.residual(features))


class TCNClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        channels: list[int],
        kernel_size: int = 3,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        blocks = []
        current = input_size
        for channel in channels:
            blocks.append(TemporalBlock(current, channel, kernel_size, dropout))
            current = channel
        self.network = nn.Sequential(*blocks)
        self.classifier = nn.Linear(current, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.network(features.transpose(1, 2))
        return self.classifier(encoded.mean(dim=2))


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, maximum_length: int = 512) -> None:
        super().__init__()
        positions = torch.arange(maximum_length).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, model_dim, 2) * (-math.log(10000.0) / model_dim)
        )
        encoding = torch.zeros(maximum_length, model_dim)
        encoding[:, 0::2] = torch.sin(positions * divisor)
        encoding[:, 1::2] = torch.cos(positions * divisor)
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.encoding[:, : features.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        model_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        feedforward_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(input_size, model_dim)
        self.position = PositionalEncoding(model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.classifier = nn.Linear(model_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.position(self.projection(features)))
        return self.classifier(encoded.mean(dim=1))


def build_model(config: dict[str, Any], num_classes: int) -> nn.Module:
    name = str(config["name"]).lower()
    if name in {"lstm", "bilstm"}:
        return LSTMClassifier(
            input_size=int(config["input_size"]),
            num_classes=num_classes,
            hidden_size=int(config.get("hidden_size", 128)),
            num_layers=int(config.get("num_layers", 2)),
            dropout=float(config.get("dropout", 0.3)),
            bidirectional=bool(config.get("bidirectional", name == "bilstm")),
        )
    if name == "tcn":
        return TCNClassifier(
            input_size=int(config["input_size"]),
            num_classes=num_classes,
            channels=[int(value) for value in config.get("channels", [128, 128, 128])],
            kernel_size=int(config.get("kernel_size", 3)),
            dropout=float(config.get("dropout", 0.25)),
        )
    if name in {"transformer", "compact_transformer"}:
        return TransformerClassifier(
            input_size=int(config["input_size"]),
            num_classes=num_classes,
            model_dim=int(config.get("model_dim", 128)),
            num_heads=int(config.get("num_heads", 4)),
            num_layers=int(config.get("num_layers", 2)),
            feedforward_dim=int(config.get("feedforward_dim", 256)),
            dropout=float(config.get("dropout", 0.2)),
        )
    raise ValueError(f"unsupported model: {name}")
