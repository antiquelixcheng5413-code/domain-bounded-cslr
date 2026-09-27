"""CTC training and evaluation loop for ordered gloss recognition."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

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
from cslr.recognition.decode import classes_to_token_ids, greedy_decode, prefix_beam_search
from cslr.recognition.gloss_sequence import GlossVocabulary
from cslr.recognition.metrics import gloss_metrics
from cslr.recognition.model import BLANK_INDEX, CTCConfig, CTCRecognizer


@dataclass
class TrainingConfig:
    epochs: int = 60
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    seed: int = 42
    device: str = "auto"
    amp: bool = True
    early_stopping_patience: int = 10
    beam_width: int = 1
    min_epochs: int = 1
    log_every: int = 1
    normalize_features: bool = True
    normalizer_sample_limit: int = 600
    # Named subset of the cached 368-dim vector, e.g. "hands+hand_deltas" (252 dims). The
    # handoff diagnostics show the full vector is close to uninformative about the gloss, and
    # the hand block is only ~34% of it.
    feature_view: str = "full"
    # frames per clip the cached features were extracted with; recorded so single-video
    # inference can reproduce the same layout
    sequence_length: int = 96
    # When a frame-level encoder produces few frames (e.g. 16 sampled images), some long gloss
    # sequences cannot be aligned at all (CTC needs output_length >= 2*target_length-1). Such
    # samples are dropped and counted instead of raising, so the same training entry point works
    # for both the 96-frame landmark cache and a short-frame CLIP cache.
    drop_unalignable_targets: bool = True
    # ReduceLROnPlateau with patience 3 was measured to decay the learning rate ~8x by epoch 22
    # and freeze training mid-run. A short warmup followed by cosine decay keeps exploring.
    lr_schedule: str = "cosine"
    warmup_epochs: int = 2
    min_learning_rate_ratio: float = 0.05

    def as_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "grad_clip": self.grad_clip,
            "seed": self.seed,
            "device": self.device,
            "amp": self.amp,
            "early_stopping_patience": self.early_stopping_patience,
            "beam_width": self.beam_width,
            "min_epochs": self.min_epochs,
            "normalize_features": self.normalize_features,
            "normalizer_sample_limit": self.normalizer_sample_limit,
            "feature_view": self.feature_view,
            "sequence_length": self.sequence_length,
            "drop_unalignable_targets": self.drop_unalignable_targets,
            "lr_schedule": self.lr_schedule,
            "warmup_epochs": self.warmup_epochs,
            "min_learning_rate_ratio": self.min_learning_rate_ratio,
        }


@dataclass
class EvaluationResult:
    metrics: dict[str, Any]
    predictions: list[list[str]] = field(default_factory=list)
    references: list[list[str]] = field(default_factory=list)
    sample_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics,
            "predictions": self.predictions,
            "references": self.references,
            "sample_ids": self.sample_ids,
        }


def learning_rate_at(
    epoch: int,
    epochs: int,
    base: float,
    warmup_epochs: int = 0,
    schedule: str = "cosine",
    min_ratio: float = 0.05,
) -> float:
    """Warmup + decay schedule, computed explicitly so it is testable.

    ``epoch`` is 1-based. Warmup is linear over whole epochs; after that ``cosine`` decays to
    ``min_ratio * base`` and ``constant`` keeps the base rate.
    """

    import math

    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if epoch < 1:
        raise ValueError("epoch must be at least 1")
    if warmup_epochs >= 1 and epoch <= warmup_epochs:
        return base * (epoch / float(warmup_epochs))
    if schedule == "constant" or epochs <= warmup_epochs:
        return base
    progress = (epoch - warmup_epochs) / float(max(epochs - warmup_epochs, 1))
    progress = min(max(progress, 0.0), 1.0)
    if schedule == "cosine":
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return base * (min_ratio + (1.0 - min_ratio) * factor)
    if schedule == "linear":
        return base * (min_ratio + (1.0 - min_ratio) * (1.0 - progress))
    raise ValueError(f"unsupported lr schedule: {schedule}")


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_torch_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "features": torch.from_numpy(batch["features"]).to(device),
        "input_lengths": torch.tensor(batch["input_lengths"], dtype=torch.long, device=device),
        "targets": pad_targets(batch["targets"], batch["target_lengths"]).to(device),
        "target_lengths": torch.tensor(batch["target_lengths"], dtype=torch.long, device=device),
        "sample_ids": batch["sample_ids"],
        "tokens": batch["tokens"],
    }


def iterate_batches(
    dataset: GlossSequenceDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> list[list[SequenceSample]]:
    """Deterministic batching (the seed is re-applied for every epoch)."""

    order = list(range(len(dataset)))
    if shuffle:
        random.Random(seed).shuffle(order)
    batches: list[list[SequenceSample]] = []
    for start in range(0, len(order), batch_size):
        batches.append([dataset[index] for index in order[start : start + batch_size]])
    return batches


def pad_targets(targets: list[int], target_lengths: list[int], blank: int = BLANK_INDEX) -> torch.Tensor:
    """Build the ``[N, S]`` target tensor torch's CTC loss expects.

    A flattened 1-D target is ambiguous to torch (it reads it as ``[N, S]`` with ``N`` equal to
    the total token count), so the batch is padded explicitly instead.
    """

    if not target_lengths:
        raise ValueError("target_lengths must not be empty")
    width = max(target_lengths)
    padded = torch.full((len(target_lengths), max(width, 1)), blank, dtype=torch.long)
    position = 0
    for row, length in enumerate(target_lengths):
        if length:
            padded[row, :length] = torch.tensor(targets[position : position + length], dtype=torch.long)
        position += length
    if position != len(targets):
        raise ValueError(
            f"targets length {len(targets)} does not match sum(target_lengths) {position}"
        )
    return padded


def ctc_loss(
    logits: torch.Tensor,
    input_lengths: torch.Tensor,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    output_lengths: torch.Tensor,
    blank: int = BLANK_INDEX,
    reduction: str = "mean",
) -> torch.Tensor:
    """CTC loss over a padded batch, with hard preconditions and an explicit reduction.

    ``logits`` is ``[B, T, C]`` and ``targets`` is ``[B, S]`` (blank padded); torch expects
    ``[T, B, C]`` log probabilities, so ``log_softmax`` and a transpose happen here.

    Two things this wrapper exists for:

    1. ``zero_infinity`` stays **off**. A sample whose target cannot fit in its frame count
       yields an infinite loss; torch would silently replace it with 0 (which corrupts the
       average and can even make the reported loss negative) instead of surfacing a data bug.
       The checks below raise instead.
    2. torch's ``reduction="mean"`` normalises by the **total target length**, not by the batch
       size, which makes the number hard to read while debugging. ``reduction="mean"`` here means
       *loss per sample* (``reduction="sum"`` divided by the batch size).

    Note the input convention, which was verified against an independent CTC forward
    implementation: ``nn.CTCLoss`` on this torch build must be given **log probabilities**
    (``log_softmax`` of the logits), not raw logits. Feeding raw logits returns the loss with the
    wrong sign (measured: -10.74 instead of +10.11 on a fixed random example), which silently
    trains the model in the wrong direction. ``tests/test_ctc_training.py`` locks this down.
    """

    if targets.numel():
        largest = int(targets.max().item())
        if largest >= logits.shape[-1]:
            raise ValueError(
                f"target id {largest} is out of range for {logits.shape[-1]} classes "
                "(the CTC blank is class 0, a vocabulary index i maps to class i + 1)"
            )
    feasible = output_lengths >= (2 * target_lengths - 1)
    if not bool(feasible.all()):
        bad = (~feasible).nonzero(as_tuple=False).flatten().tolist()
        details = ", ".join(
            f"row {row}: {int(output_lengths[row])} frames < 2*{int(target_lengths[row])}-1"
            for row in bad[:5]
        )
        raise ValueError(
            "CTC alignment is impossible for some samples (needs output_length >= 2*target_length-1): "
            f"{details}"
        )

    log_probs = torch.log_softmax(logits.float(), dim=-1).transpose(0, 1)
    per_sample = nn.CTCLoss(blank=blank, reduction="none", zero_infinity=False)(
        log_probs, targets, output_lengths, target_lengths
    )
    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    if reduction == "mean":
        return per_sample.mean()
    raise ValueError(f"unsupported reduction: {reduction}")


def ctc_loss_per_target_token(
    logits: torch.Tensor,
    input_lengths: torch.Tensor,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    output_lengths: torch.Tensor,
    blank: int = BLANK_INDEX,
) -> torch.Tensor:
    """torch's own ``mean`` convention (divide by total target length) for comparability."""

    per_sample = ctc_loss(
        logits, input_lengths, targets, target_lengths, output_lengths, blank, reduction="none"
    )
    total = target_lengths.clamp(min=1).sum()
    return per_sample.sum() / total


def decode_batch(log_probs: np.ndarray, lengths: list[int], beam_width: int) -> tuple[list[list[int]], int, int]:
    """Decode one batch; returns token id sequences plus blank/step counters."""

    decoded: list[list[int]] = []
    blank_steps = 0
    total_steps = 0
    for row, length in enumerate(lengths):
        steps = log_probs[row, :length]
        if beam_width > 1:
            result = prefix_beam_search(steps, beam_width=beam_width)
        else:
            result = greedy_decode(steps)
        decoded.append(classes_to_token_ids(result.classes))
        total_steps += steps.shape[0]
        blank_steps += int((steps.argmax(axis=1) == BLANK_INDEX).sum())
    return decoded, blank_steps, total_steps


@torch.no_grad()
def evaluate(
    model: CTCRecognizer,
    dataset: GlossSequenceDataset,
    vocabulary: GlossVocabulary,
    batch_size: int,
    device: torch.device,
    beam_width: int = 1,
    seed: int = 42,
    amp: bool = False,
) -> EvaluationResult:
    model.eval()
    predictions: list[list[str]] = []
    references: list[list[str]] = []
    sample_ids: list[str] = []
    blank_steps = 0
    total_steps = 0
    total_loss = 0.0
    total_samples = 0

    for samples in iterate_batches(dataset, batch_size, shuffle=False, seed=seed):
        batch = to_torch_batch(collate_samples(samples), device)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(batch["features"], batch["input_lengths"])
        output_lengths = model.output_lengths(batch["input_lengths"])
        loss = ctc_loss(
            logits, batch["input_lengths"], batch["targets"], batch["target_lengths"], output_lengths
        )
        total_loss += float(loss.item()) * len(samples)
        total_samples += len(samples)
        # decoding needs explicit log probabilities; the loss above intentionally gets raw logits
        log_probs = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
        decoded, batch_blank, batch_steps = decode_batch(
            log_probs, output_lengths.cpu().tolist(), beam_width
        )
        blank_steps += batch_blank
        total_steps += batch_steps
        for row, token_ids in enumerate(decoded):
            predictions.append(vocabulary.decode(token_ids))
            references.append(batch["tokens"][row])
            sample_ids.append(batch["sample_ids"][row])

    metrics = gloss_metrics(
        references,
        predictions,
        vocabulary_size=vocabulary.size,
        blank_steps=blank_steps,
        total_steps=total_steps,
    )
    metrics["loss"] = total_loss / total_samples if total_samples else 0.0
    return EvaluationResult(
        metrics=metrics,
        predictions=predictions,
        references=references,
        sample_ids=sample_ids,
    )


@dataclass
class TrainResult:
    checkpoint_path: Path
    history: list[dict[str, Any]]
    best_epoch: int
    best_wer: float
    validation: EvaluationResult
    train_samples: int
    validation_samples: int
    device: str
    vocabulary_size: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": str(self.checkpoint_path),
            "best_epoch": self.best_epoch,
            "best_wer": self.best_wer,
            "train_samples": self.train_samples,
            "validation_samples": self.validation_samples,
            "device": self.device,
            "vocabulary_size": self.vocabulary_size,
            "history": self.history,
            "validation_metrics": self.validation.metrics,
        }


def _drop_unalignable(
    records: list[SampleRecord],
    vocabulary: GlossVocabulary,
    feature_root: Path,
    chunks: int = 8,
) -> tuple[list[SampleRecord], list[str]]:
    """Drop samples whose target cannot align with the cached frame count.

    CTC needs ``frames >= 2*target_length - 1``. A short-frame cache (16 sampled CLIP images)
    cannot hold a 10-token gloss, and the loss would otherwise raise on that batch. The frame
    count is read from the cached arrays, so nothing is assumed about the encoder.
    """

    import numpy as np  # local import: keeps the module import-time deps unchanged

    kept: list[SampleRecord] = []
    dropped: list[str] = []
    for record in records:
        target_length = len(vocabulary.encode(record.label))
        path = feature_root / f"{record.sample_id}.npy"
        if not path.exists():
            dropped.append(record.sample_id)
            continue
        frames = int(np.load(path, mmap_mode="r").shape[0])
        if target_length and frames < 2 * target_length - 1:
            dropped.append(record.sample_id)
        else:
            kept.append(record)
    return kept, dropped


def train_ctc(
    manifest_path: Path,
    feature_root: Path,
    vocabulary: GlossVocabulary,
    model_config: CTCConfig,
    training_config: TrainingConfig,
    output_path: Path,
    limit_train: int | None = None,
    limit_validation: int | None = None,
    present_only: bool = False,
) -> TrainResult:
    """Train the CTC gloss recognizer. Nothing here touches the test split.

    ``present_only`` skips records whose feature file has not been extracted yet, which is what
    makes iterative training possible while the (hours long) extraction is still running. The
    dropped ids are reported in the result so a run can never silently pretend to use everything.
    """

    records = load_records(manifest_path)
    train_records = split_records(records, "train")
    validation_records = split_records(records, "validation")
    if not train_records or not validation_records:
        raise ValueError("manifest must contain train and validation records")
    skipped_train: list[str] = []
    skipped_validation: list[str] = []
    if present_only:
        train_records, skipped_train = filter_present(train_records, feature_root)
        validation_records, skipped_validation = filter_present(validation_records, feature_root)
        if not train_records:
            raise ValueError(f"no extracted train features under {feature_root}")
        if not validation_records:
            raise ValueError(f"no extracted validation features under {feature_root}")
    if limit_train:
        train_records = train_records[:limit_train]
    if limit_validation:
        validation_records = validation_records[:limit_validation]

    random.seed(training_config.seed)
    np.random.seed(training_config.seed)
    torch.manual_seed(training_config.seed)

    device = resolve_device(training_config.device)
    model = CTCRecognizer(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    use_amp = training_config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    dropped_train: list[str] = []
    dropped_validation: list[str] = []
    if training_config.drop_unalignable_targets:
        train_records, dropped_train = _drop_unalignable(train_records, vocabulary, feature_root)
        validation_records, dropped_validation = _drop_unalignable(
            validation_records, vocabulary, feature_root
        )
        if not train_records or not validation_records:
            raise ValueError("every sample was dropped as unalignable; use a longer frame cache")

    train_dataset = GlossSequenceDataset(
        train_records, feature_root, vocabulary, feature_view=training_config.feature_view
    )
    validation_dataset = GlossSequenceDataset(
        validation_records, feature_root, vocabulary, feature_view=training_config.feature_view
    )
    normalizer: FeatureNormalizer | None = None
    if training_config.normalize_features:
        normalizer = build_normalizer(
            train_records,
            feature_root,
            sample_limit=training_config.normalizer_sample_limit,
            feature_view=training_config.feature_view,
        )
        train_dataset = GlossSequenceDataset(
            train_records, feature_root, vocabulary, normalizer, feature_view=training_config.feature_view
        )
        validation_dataset = GlossSequenceDataset(
            validation_records,
            feature_root,
            vocabulary,
            normalizer,
            feature_view=training_config.feature_view,
        )

    history: list[dict[str, Any]] = []
    best_wer = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] = {}
    epochs_without_improvement = 0
    started = time.perf_counter()

    for epoch in range(1, training_config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        seen = 0
        current_lr = learning_rate_at(
            epoch,
            training_config.epochs,
            training_config.learning_rate,
            warmup_epochs=training_config.warmup_epochs,
            schedule=training_config.lr_schedule,
            min_ratio=training_config.min_learning_rate_ratio,
        )
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        for samples in iterate_batches(
            train_dataset, training_config.batch_size, shuffle=True, seed=training_config.seed + epoch
        ):
            batch = to_torch_batch(collate_samples(samples), device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(batch["features"], batch["input_lengths"])
                output_lengths = model.output_lengths(batch["input_lengths"])
                loss = ctc_loss(
                    logits,
                    batch["input_lengths"],
                    batch["targets"],
                    batch["target_lengths"],
                    output_lengths,
                )
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.grad_clip)
                optimizer.step()
            epoch_loss += float(loss.item()) * len(samples)
            seen += len(samples)

        validation = evaluate(
            model,
            validation_dataset,
            vocabulary,
            training_config.batch_size,
            device,
            beam_width=training_config.beam_width,
            seed=training_config.seed,
            amp=use_amp,
        )
        wer = float(validation.metrics["wer"])
        record = {
            "epoch": epoch,
            "train_loss": epoch_loss / seen if seen else 0.0,
            "validation_loss": validation.metrics["loss"],
            "wer": wer,
            "cer": validation.metrics["cer"],
            "sequence_exact_match": validation.metrics["sequence_exact_match"],
            "empty_hypothesis_rate": validation.metrics["empty_hypothesis_rate"],
            "blank_ratio": validation.metrics["blank_ratio"],
            "vocabulary_utilization": validation.metrics.get("vocabulary_utilization", 0.0),
            "learning_rate": current_lr,
        }
        history.append(record)
        if epoch % training_config.log_every == 0 or epoch == 1:
            print(json.dumps(record, ensure_ascii=False), flush=True)

        if wer < best_wer:
            best_wer = wer
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if (
                epochs_without_improvement >= training_config.early_stopping_patience
                and epoch >= training_config.min_epochs
            ):
                print(json.dumps({"early_stop": True, "epoch": epoch}), flush=True)
                break

    if not best_state:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        best_epoch = len(history)
    model.load_state_dict(best_state)
    final_validation = evaluate(
        model,
        validation_dataset,
        vocabulary,
        training_config.batch_size,
        device,
        beam_width=training_config.beam_width,
        seed=training_config.seed,
        amp=use_amp,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": model_config.as_dict(),
            "training_config": training_config.as_dict(),
            "vocabulary": list(vocabulary.tokens),
            "vocabulary_counts": dict(vocabulary.counts),
            "vocabulary_config": vocabulary.config.as_dict(),
            "feature_normalizer": normalizer.as_dict() if normalizer else None,
            "best_epoch": best_epoch,
            "best_wer": best_wer,
            "state_dict": best_state,
            "history": history,
        },
        output_path,
    )
    (output_path.with_suffix(".history.json")).write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "checkpoint": str(output_path),
                "best_epoch": best_epoch,
                "best_wer": best_wer,
                "elapsed_seconds": round(time.perf_counter() - started, 2),
                "device": str(device),
                "train_samples": len(train_records),
                "validation_samples": len(validation_records),
                "normalized_features": normalizer is not None,
                "dropped_unalignable_train": len(dropped_train),
                "dropped_unalignable_validation": len(dropped_validation),
                "skipped_train_without_features": len(skipped_train),
                "skipped_validation_without_features": len(skipped_validation),
                "present_only": present_only,
                "test_split_read": False,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return TrainResult(
        checkpoint_path=output_path,
        history=history,
        best_epoch=best_epoch,
        best_wer=best_wer,
        validation=final_validation,
        train_samples=len(train_records),
        validation_samples=len(validation_records),
        device=str(device),
        vocabulary_size=vocabulary.size,
    )
