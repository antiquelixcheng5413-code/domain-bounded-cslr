"""Cross-modal contrastive pretraining: align video clips with their gloss labels.

Sections 20/21 found the cached VL48 features carry strong content-gloss information
(mean-pool + linear reaches 0.68 macro-AUC) that the CTC model never exploits
(strong-gloss dev recall 0.0). This module trains a lightweight alignment head on
the *train* split only:

    video  -> ContrastiveEncoder (LayerNorm -> projection -> length-aware mean pool)
    gloss  -> GlossEmbedding per vocabulary token

and pulls each clip toward the gloss tokens it contains (multi-label InfoNCE over
the vocabulary, in-batch negatives). The encoder frontend (``normalize.*``,
``projection.*``) is deliberately the same subgraph as :class:`CTCRecognizer`'s, so
``copy_frontend_weights`` can warm-start CTC training with it.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from cslr.recognition.dataset import (
    FeatureNormalizer,
    GlossSequenceDataset,
    SequenceSample,
    build_normalizer,
    collate_samples,
    filter_present,
    load_records,
    split_records,
)
from cslr.recognition.gloss_sequence import GlossVocabulary
from cslr.recognition.training import (
    iterate_batches,
    learning_rate_at,
    resolve_device,
    to_torch_batch,
)

UNKNOWN_INDEX = 0


@dataclass(frozen=True)
class ContrastiveConfig:
    input_size: int = 2048
    projection_size: int = 256
    embedding_size: int = 256
    temperature: float = 0.07
    epochs: int = 40
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    seed: int = 42
    device: str = "auto"
    amp: bool = True
    log_every: int = 1
    lr_schedule: str = "cosine"
    warmup_epochs: int = 2
    min_learning_rate_ratio: float = 0.05

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_size": self.input_size,
            "projection_size": self.projection_size,
            "embedding_size": self.embedding_size,
            "temperature": self.temperature,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "grad_clip": self.grad_clip,
            "seed": self.seed,
            "device": self.device,
            "amp": self.amp,
            "lr_schedule": self.lr_schedule,
            "warmup_epochs": self.warmup_epochs,
            "min_learning_rate_ratio": self.min_learning_rate_ratio,
        }


class ContrastiveEncoder(nn.Module):
    """LayerNorm -> projection -> length-aware mean pool -> L2 norm.

    ``normalize`` and ``projection`` are structurally identical to
    :class:`CTCRecognizer`'s frontend, so their weights transfer directly.
    """

    def __init__(self, input_size: int, projection_size: int = 256) -> None:
        super().__init__()
        self.normalize = nn.LayerNorm(input_size)
        self.projection = nn.Sequential(
            nn.Linear(input_size, projection_size),
            nn.GELU(),
            nn.Dropout(0.3),
        )

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        hidden = self.projection(self.normalize(features))
        pooled = mean_pool_hidden(hidden, lengths)
        return F.normalize(pooled, dim=-1)


class GlossEmbedding(nn.Module):
    """Normalised embedding table over the ordered gloss vocabulary (index 0 = <unk>)."""

    def __init__(self, vocabulary_size: int, embedding_size: int = 256) -> None:
        super().__init__()
        self.table = nn.Embedding(vocabulary_size, embedding_size)

    def normalized_embeddings(self) -> torch.Tensor:
        return F.normalize(self.table.weight, dim=-1)


def mean_pool_hidden(hidden: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Average over the real timesteps only; padded rows never contribute."""

    steps = hidden.shape[1]
    positions = torch.arange(steps, device=hidden.device).unsqueeze(0)
    mask = (positions < lengths.unsqueeze(1)).unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    return summed / lengths.unsqueeze(1).clamp(min=1).to(hidden.dtype)


def positive_gloss_ids(samples: list[SequenceSample]) -> list[list[int]]:
    """Per-sample set of positive gloss ids, excluding <unk> (index 0)."""

    rows: list[list[int]] = []
    for sample in samples:
        rows.append(sorted({token_id for token_id in sample.token_ids if token_id != UNKNOWN_INDEX}))
    return rows


def contrastive_loss(
    anchors: torch.Tensor,
    positive_ids: list[list[int]],
    gloss_embeddings: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Multi-label InfoNCE over the vocabulary, mean over each sample's positive tokens.

    ``anchors`` are the L2-normalised video embeddings ``[B, d]`` and
    ``gloss_embeddings`` the normalised vocabulary ``[V, d]``. Every positive token of a
    clip is pulled towards it; all other vocabulary tokens act as negatives via the
    shared softmax.
    """

    logits = anchors @ gloss_embeddings.t() / temperature
    log_probs = F.log_softmax(logits, dim=-1)
    per_sample: list[torch.Tensor] = []
    for row, ids in enumerate(positive_ids):
        if not ids:
            continue
        per_sample.append(-log_probs[row, ids].mean())
    if not per_sample:
        return torch.zeros((), device=anchors.device)
    return torch.stack(per_sample).mean()


def copy_frontend_weights(
    pretrain_state: dict[str, torch.Tensor], ctc_model: nn.Module
) -> tuple[list[str], list[str], list[str]]:
    """Copy the shared ``normalize.*`` / ``projection.*`` keys into a CTCRecognizer.

    Only keys whose shape matches are copied; a mismatched frontend (different input or
    projection size) is reported in ``skipped`` instead of raising. Returns
    ``(missing, unexpected, skipped_shape_mismatch)`` for the caller to log; the CTC model's
    LSTM and classifier stay randomly initialised.
    """

    state = ctc_model.state_dict()
    mapping: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, value in pretrain_state.items():
        if not (key.startswith("normalize.") or key.startswith("projection.")):
            continue
        if key not in state or state[key].shape != value.shape:
            skipped.append(key)
            continue
        mapping[key] = value
    missing, unexpected = ctc_model.load_state_dict(mapping, strict=False)
    return sorted(missing), sorted(unexpected), sorted(skipped)


@dataclass
class PretrainResult:
    checkpoint_path: Path
    history: list[dict[str, Any]]
    train_samples: int
    device: str
    vocabulary_size: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": str(self.checkpoint_path),
            "train_samples": self.train_samples,
            "device": self.device,
            "vocabulary_size": self.vocabulary_size,
            "history": self.history,
        }


def pretrain_contrastive(
    manifest_path: Path,
    feature_root: Path,
    vocabulary: GlossVocabulary,
    config: ContrastiveConfig,
    output_path: Path,
    present_only: bool = False,
    normalizer_sample_limit: int = 600,
    limit_train: int | None = None,
) -> PretrainResult:
    """Train the contrastive alignment head on the train split only (test never read)."""

    records = load_records(manifest_path)
    train_records = split_records(records, "train")
    if not train_records:
        raise ValueError("manifest contains no train records")
    if limit_train:
        train_records = train_records[:limit_train]
    skipped: list[str] = []
    if present_only:
        train_records, skipped = filter_present(train_records, feature_root)
        if not train_records:
            raise ValueError(f"no extracted train features under {feature_root}")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = resolve_device(config.device)
    encoder = ContrastiveEncoder(config.input_size, config.projection_size).to(device)
    gloss_embedding = GlossEmbedding(vocabulary.size, config.embedding_size).to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(gloss_embedding.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    normalizer: FeatureNormalizer | None = None
    if normalizer_sample_limit:
        normalizer = build_normalizer(
            train_records,
            feature_root,
            sample_limit=normalizer_sample_limit,
            feature_view="full",
        )
    dataset = GlossSequenceDataset(
        train_records, feature_root, vocabulary, normalizer, feature_view="full"
    )

    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        encoder.train()
        gloss_embedding.train()
        epoch_loss = 0.0
        epoch_hits = 0
        epoch_samples = 0
        current_lr = learning_rate_at(
            epoch,
            config.epochs,
            config.learning_rate,
            warmup_epochs=config.warmup_epochs,
            schedule=config.lr_schedule,
            min_ratio=config.min_learning_rate_ratio,
        )
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        for samples in iterate_batches(
            dataset, config.batch_size, shuffle=True, seed=config.seed + epoch
        ):
            batch = to_torch_batch(collate_samples(samples), device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                anchors = encoder(batch["features"], batch["input_lengths"])
                embeddings = gloss_embedding.normalized_embeddings()
                loss = contrastive_loss(
                    anchors,
                    positive_gloss_ids(samples),
                    embeddings,
                    config.temperature,
                )
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(gloss_embedding.parameters()),
                    config.grad_clip,
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(gloss_embedding.parameters()),
                    config.grad_clip,
                )
                optimizer.step()
            epoch_loss += float(loss.item()) * len(samples)
            epoch_samples += len(samples)
            with torch.no_grad():
                best = anchors.argmax(dim=-1)
                for row, ids in enumerate(positive_gloss_ids(samples)):
                    if ids and int(best[row]) in ids:
                        epoch_hits += 1

        record = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(epoch_samples, 1),
            "batch_hit_rate": (epoch_hits / max(epoch_samples, 1)) if epoch_samples else 0.0,
            "learning_rate": current_lr,
        }
        history.append(record)
        if epoch % config.log_every == 0 or epoch == 1:
            print(json.dumps(record, ensure_ascii=False), flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "pretrain_config": config.as_dict(),
            "vocabulary": list(vocabulary.tokens),
            "vocabulary_counts": dict(vocabulary.counts),
            "vocabulary_config": vocabulary.config.as_dict(),
            "feature_normalizer": normalizer.as_dict() if normalizer else None,
            "state_dict": encoder.state_dict(),
            "gloss_embedding": gloss_embedding.state_dict(),
            "history": history,
        },
        output_path,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "checkpoint": str(output_path),
                "train_samples": len(train_records),
                "device": str(device),
                "elapsed_seconds": round(time.perf_counter() - started, 2),
                "skipped_without_features": len(skipped),
                "test_split_read": False,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return PretrainResult(
        checkpoint_path=output_path,
        history=history,
        train_samples=len(train_records),
        device=str(device),
        vocabulary_size=vocabulary.size,
    )
