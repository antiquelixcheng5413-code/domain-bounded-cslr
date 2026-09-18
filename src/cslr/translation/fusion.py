"""SpaMo-style spatial-motion multimodal fusion.

Concatenates per-modality tokens, adds positional encoding, then runs a
Transformer encoder. Any modality can be disabled via config so the fusion
supports single/dual/tri-modal ablations. Part 2 transition-mask input is
accepted if provided but never fabricated.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from cslr.translation.encoders import BaseSequenceEncoder, build_encoder


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ModalityProjection(nn.Module):
    """Project one modality's tokens to hidden_dim regardless of input dim."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class TokenPooling(nn.Module):
    """SpaMo-style fixed-size pooling: compress variable-length fused tokens to a
    fixed number of learnable-query tokens via cross-attention.

    ``num_queries`` is the constant output token count; the Decoder therefore sees a
    fixed-length visual context regardless of input video length.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_queries: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.zeros(1, num_queries, hidden_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.pool = nn.TransformerDecoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch = memory.shape[0]
        queries = self.queries.expand(batch, -1, -1)
        memory_key_padding = None if memory_mask is None else ~memory_mask
        pooled = self.pool(queries, memory, memory_key_padding_mask=memory_key_padding)
        return self.norm(pooled)


class LightSpaMoFusion(nn.Module):
    """SpaMo-style multimodal fusion with configurable modality participation.

    ``feature_dim`` maps each modality to the raw input feature width (e.g.
    landmark=368, rgb=hidden_dim, motion=hidden_dim). If omitted, a sensible
    default is used per modality.
    """

    def __init__(
        self,
        modalities: tuple[str, ...],
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        feedforward_dim: int,
        dropout: float = 0.0,
        feature_dim: dict[str, int] | None = None,
        encoders: dict[str, BaseSequenceEncoder] | None = None,
        max_seq_len: int = 2048,
        num_pool_tokens: int | None = None,
    ) -> None:
        super().__init__()
        self.modalities = list(modalities)
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_pool_tokens = num_pool_tokens or 0

        self.encoders = nn.ModuleDict()
        self.projectors = nn.ModuleDict()
        for mod in self.modalities:
            if encoders is not None and mod in encoders:
                self.encoders[mod] = encoders[mod]
                input_dim = getattr(encoders[mod], "hidden_dim", hidden_dim)
            else:
                input_dim = (feature_dim or {}).get(
                    mod, 368 if mod == "landmark" else hidden_dim
                )
                self.encoders[mod] = build_encoder(
                    mod, input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout
                )
            self.projectors[mod] = ModalityProjection(
                hidden_dim, hidden_dim, dropout
            )

        self.positional = PositionalEncoding(hidden_dim, max_len=max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.pooling: TokenPooling | None = None
        if self.num_pool_tokens > 0:
            self.pooling = TokenPooling(
                hidden_dim=hidden_dim,
                num_queries=self.num_pool_tokens,
                num_heads=num_heads,
                dropout=dropout,
            )

    def forward(
        self,
        features: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor] | None = None,
        transition_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return (visual_tokens, visual_mask, actual_token_count)."""
        pad_mask = masks or {
            mod: features[mod].new_ones(features[mod].shape[:2], dtype=torch.bool)
            for mod in features
        }
        parts: list[torch.Tensor] = []
        part_masks: list[torch.Tensor] = []

        for mod in self.modalities:
            if mod not in features:
                continue
            raw = features[mod]
            encoded, _enc_mask = self.encoders[mod].encode(raw, pad_mask.get(mod))
            proj = self.projectors[mod](encoded)
            mod_mask = encoded.new_ones(encoded.shape[:2], dtype=torch.bool)
            if masks is not None and mod in masks:
                mod_mask = masks[mod][:, : encoded.shape[1]]
            parts.append(proj)
            part_masks.append(mod_mask)

        if not parts:
            raise ValueError("no active modalities provided to fusion")

        concat = torch.cat(parts, dim=1)
        concat_mask = torch.cat(part_masks, dim=1)
        concat = self.positional(concat)

        key_padding = ~concat_mask
        attended = self.transformer(
            concat,
            src_key_padding_mask=key_padding if key_padding.any() else None,
        )
        attended = self.output_norm(attended)

        # SpaMo-style fixed-size pooling: compress variable-length fused tokens
        # to ``num_pool_tokens`` learnable-query tokens (constant across samples).
        if self.pooling is not None:
            pooled = self.pooling(attended, concat_mask)
            pooled_mask = pooled.new_ones(pooled.shape[:2], dtype=torch.bool)
            return pooled, pooled_mask, int(pooled.shape[1])

        # Per-sample actual token count (max across batch) for reporting; the
        # full batch mask is still returned for the decoder.
        attended_len = int(concat_mask.sum(dim=1).max().item())
        return attended, concat_mask, attended_len