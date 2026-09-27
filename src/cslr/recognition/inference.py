"""Load a trained CTC checkpoint and decode a split end to end.

This is the "loadable recognizer" deliverable: it reads a checkpoint produced by
``python -m cslr.recognition train``, rebuilds the model, vocabulary and feature normalizer from
the checkpoint itself, and writes per-sample predictions plus metrics. It never reads the frozen
test split.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cslr.recognition.dataset import (
    FeatureNormalizer,
    GlossSequenceDataset,
    collate_samples,
    feature_view_indices,
    filter_present,
    load_records,
    split_records,
)
from cslr.recognition.decode import classes_to_token_ids, greedy_decode, prefix_beam_search
from cslr.recognition.gloss_sequence import GlossSequenceConfig, GlossVocabulary
from cslr.recognition.metrics import gloss_metrics
from cslr.recognition.model import BLANK_INDEX, CTCConfig, CTCRecognizer, ctc_config_from_dict
from cslr.recognition.training import ctc_loss, iterate_batches, resolve_device, to_torch_batch

FROZEN_SPLIT_ERROR = (
    "split 'test' is frozen: the official 500-sample test split must not be read, inferred on, "
    "or tuned against during this phase"
)


def load_recognizer(checkpoint_path: Path, device: str = "auto"):
    """Rebuild model + vocabulary + normalizer from a checkpoint."""

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint is missing: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = ctc_config_from_dict(dict(payload["model_config"]))
    model = CTCRecognizer(model_config)
    model.load_state_dict(payload["state_dict"])
    torch_device = resolve_device(device)
    model.to(torch_device)
    model.eval()
    raw_config = dict(payload.get("vocabulary_config") or {})
    vocabulary_config = GlossSequenceConfig(
        keep_punctuation=bool(raw_config.get("keep_punctuation", True)),
        keep_numeric_tokens=bool(raw_config.get("keep_numeric_tokens", True)),
        strip_variant_numbering=bool(raw_config.get("strip_variant_numbering", True)),
        strip_annotations=bool(raw_config.get("strip_annotations", True)),
        target_unit=str(raw_config.get("target_unit", "token")),
    )
    vocabulary_config.validate()
    vocabulary = GlossVocabulary(
        tokens=tuple(str(token) for token in payload["vocabulary"]),
        counts={str(k): int(v) for k, v in dict(payload.get("vocabulary_counts") or {}).items()},
        config=vocabulary_config,
    )
    normalizer_payload = payload.get("feature_normalizer")
    normalizer = FeatureNormalizer.from_dict(normalizer_payload) if normalizer_payload else None
    training_config = dict(payload.get("training_config") or {})
    return model, vocabulary, normalizer, training_config, torch_device


def evaluate_checkpoint(
    checkpoint_path: Path,
    manifest_path: Path,
    feature_root: Path,
    split: str,
    output_path: Path,
    beam_width: int = 1,
    limit: int | None = None,
    batch_size: int = 32,
    device: str = "auto",
    present_only: bool = True,
) -> dict[str, Any]:
    if split == "test":
        raise ValueError(FROZEN_SPLIT_ERROR)
    split_name = "validation" if split == "dev" else split
    model, vocabulary, normalizer, training_config, torch_device = load_recognizer(
        checkpoint_path, device
    )
    records = load_records(manifest_path)
    selected = split_records(records, split_name)
    if not selected:
        raise ValueError(f"manifest has no records for split {split_name}")
    skipped: list[str] = []
    if present_only:
        selected, skipped = filter_present(selected, feature_root)
        if not selected:
            raise ValueError(f"no extracted features for split {split_name} under {feature_root}")
    if limit:
        selected = selected[:limit]

    feature_view = str(training_config.get("feature_view", "full"))
    dataset = GlossSequenceDataset(
        selected, feature_root, vocabulary, normalizer, feature_view=feature_view
    )
    started = time.perf_counter()
    predictions: list[list[str]] = []
    references: list[list[str]] = []
    sample_ids: list[str] = []
    blank_steps = 0
    total_steps = 0
    total_loss = 0.0
    with torch.no_grad():
        for samples in iterate_batches(dataset, batch_size, shuffle=False, seed=0):
            batch = to_torch_batch(collate_samples(samples), torch_device)
            logits = model(batch["features"], batch["input_lengths"])
            output_lengths = model.output_lengths(batch["input_lengths"])
            log_probs = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
            loss = ctc_loss(
                logits,
                batch["input_lengths"],
                batch["targets"],
                batch["target_lengths"],
                output_lengths,
            )
            total_loss += float(loss.item()) * len(samples)
            for row, length in enumerate(output_lengths.cpu().tolist()):
                steps = log_probs[row, :length]
                result = (
                    prefix_beam_search(steps, beam_width=beam_width)
                    if beam_width > 1
                    else greedy_decode(steps)
                )
                predictions.append(vocabulary.decode(classes_to_token_ids(result.classes)))
                references.append(batch["tokens"][row])
                sample_ids.append(batch["sample_ids"][row])
                total_steps += steps.shape[0]
                blank_steps += int((steps.argmax(axis=1) == BLANK_INDEX).sum())

    metrics = gloss_metrics(
        references,
        predictions,
        vocabulary_size=vocabulary.size,
        blank_steps=blank_steps,
        total_steps=total_steps,
    )
    metrics["loss"] = total_loss / max(len(sample_ids), 1)
    elapsed = time.perf_counter() - started
    payload = {
        "checkpoint": str(checkpoint_path),
        "manifest": str(manifest_path),
        "feature_root": str(feature_root),
        "feature_view": feature_view,
        "split": split_name,
        "samples": len(sample_ids),
        "beam_width": beam_width,
        "device": str(torch_device),
        "skipped_without_features": len(skipped),
        "elapsed_seconds": round(elapsed, 2),
        "latency_ms_per_sample": round(1000 * elapsed / max(len(sample_ids), 1), 2),
        "test_split_read": False,
        "metrics": metrics,
        "predictions": [
            {
                "sample_id": sample_ids[index],
                "reference": references[index],
                "prediction": predictions[index],
            }
            for index in range(len(sample_ids))
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def collate_samples_local(samples):
    from cslr.recognition.dataset import collate_samples

    return collate_samples(samples)


def format_prediction(tokens: list[str], separator: str = "/") -> str:
    """Human-readable gloss string, skipping the unknown marker."""

    return separator.join(token for token in tokens if token != "<unk>")


def predict_video(
    checkpoint_path: Path,
    video_path: Path,
    beam_width: int = 1,
    device: str = "auto",
    sequence_length: int | None = None,
) -> dict[str, Any]:
    """Full inference path: video file -> landmarks -> CTC decode -> gloss tokens.

    The landmark extractor must produce the same layout the checkpoint was trained on, so the
    frame count defaults to the training-time value (96 unless the checkpoint says otherwise).
    """

    model, vocabulary, normalizer, training_config, torch_device = load_recognizer(
        checkpoint_path, device
    )
    if sequence_length is None:
        # the checkpoint does not record its frame count directly; 96 is the Part 4 default
        sequence_length = int(training_config.get("sequence_length", 96))
    from cslr.features.extractor import MediaPipeHolisticExtractor

    started = time.perf_counter()
    extraction = MediaPipeHolisticExtractor(sequence_length=sequence_length).extract(video_path)
    extraction_ms = (time.perf_counter() - started) * 1000
    features = extraction.features
    feature_view = str(training_config.get("feature_view", "full"))
    columns = feature_view_indices(feature_view)
    if columns is not None:
        features = features[:, columns]
    if normalizer is not None:
        features = normalizer.apply(features)
    if features.shape[0] == 0:
        raise ValueError(f"no frames could be extracted from {video_path}")

    tensor = torch.from_numpy(features[None].astype(np.float32)).to(torch_device)
    lengths = torch.tensor([features.shape[0]], dtype=torch.long, device=torch_device)
    decode_started = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor, lengths)
        output_lengths = model.output_lengths(lengths)
        steps = torch.log_softmax(logits.float(), dim=-1)[0, : int(output_lengths[0])].cpu().numpy()
    result = (
        prefix_beam_search(steps, beam_width=beam_width) if beam_width > 1 else greedy_decode(steps)
    )
    tokens = vocabulary.decode(classes_to_token_ids(result.classes))
    decode_ms = (time.perf_counter() - decode_started) * 1000
    return {
        "video": str(video_path),
        "checkpoint": str(checkpoint_path),
        "feature_view": feature_view,
        "sequence_length": sequence_length,
        "source_frames": extraction.source_frames,
        "quality_accepted": extraction.quality.accepted,
        "quality_valid_ratio": extraction.quality.valid_ratio,
        "gloss_tokens": tokens,
        "gloss": format_prediction(tokens),
        "beam_width": beam_width,
        "latency_ms": {"extraction": round(extraction_ms, 2), "decode": round(decode_ms, 2)},
    }
