from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional
from torch.utils.data import DataLoader

from cslr.config import load_yaml
from cslr.data.ctc import CTCDataset, CTCVocabulary, collate_ctc_samples, split_ctc_tokens
from cslr.data.manifest import read_manifest, validate_manifest
from cslr.evaluation.ctc import EditCounts, corpus_wer, ctc_greedy_decode, token_edit_counts
from cslr.inference.ctc import CTCOnnxRecognizer
from cslr.models.ctc import build_ctc_model, build_legacy_ctc_model


@dataclass(frozen=True)
class CTCPredictionRecord:
    split: str
    sample_id: str
    reference: list[str]
    hypothesis: list[str]
    counts: EditCounts
    path_score: float
    inference_ms: float

    def as_row(self) -> dict[str, object]:
        return {
            "split": self.split,
            "sample_id": self.sample_id,
            "reference": " ".join(self.reference),
            "hypothesis": " ".join(self.hypothesis),
            "substitutions": self.counts.substitutions,
            "deletions": self.counts.deletions,
            "insertions": self.counts.insertions,
            "reference_tokens": self.counts.reference_tokens,
            "errors": self.counts.errors,
            "path_score": self.path_score,
            "inference_ms": self.inference_ms,
        }


def _collate_for_torch(samples: list[object]) -> dict[str, object]:
    batch = collate_ctc_samples(samples)  # type: ignore[arg-type]
    batch["features"] = torch.from_numpy(batch["features"])  # type: ignore[arg-type]
    batch["targets"] = torch.from_numpy(batch["targets"])  # type: ignore[arg-type]
    batch["input_lengths"] = torch.from_numpy(batch["input_lengths"])  # type: ignore[arg-type]
    batch["target_lengths"] = torch.from_numpy(batch["target_lengths"])  # type: ignore[arg-type]
    return batch


def _make_loader(
    dataset: CTCDataset,
    batch_size: int,
    shuffle: bool,
) -> DataLoader[dict[str, object]]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate_for_torch,
    )


def _latency_percentiles(latencies: list[float]) -> dict[str, float]:
    if not latencies:
        return {"p50": 0.0, "p95": 0.0}

    def percentile(value: float) -> float:
        index = max(0, int(np.ceil(value * len(latencies))) - 1)
        return latencies[index]

    return {"p50": percentile(0.5), "p95": percentile(0.95)}


def _evaluate_model(
    model: nn.Module,
    loader: DataLoader[dict[str, object]],
    vocabulary: CTCVocabulary,
    split: str,
    device: torch.device,
) -> tuple[dict[str, object], list[CTCPredictionRecord]]:
    import time

    model.eval()
    records: list[CTCPredictionRecord] = []
    index_to_token = vocabulary.index_to_token
    with torch.no_grad():
        for batch in loader:
            features = batch["features"]
            if not isinstance(features, torch.Tensor):
                raise TypeError("CTC batch features must be a tensor")
            started = time.perf_counter()
            logits = model(features.to(device)).cpu().numpy()
            batch_ms = (time.perf_counter() - started) * 1000
            sample_ids = batch["sample_ids"]
            references = batch["reference_tokens"]
            if not isinstance(sample_ids, list) or not isinstance(references, list):
                raise TypeError("CTC batch metadata is invalid")
            for sample_id, reference, sample_logits in zip(
                sample_ids, references, logits, strict=False
            ):
                if not isinstance(sample_id, str) or not isinstance(reference, list):
                    raise TypeError("CTC sample metadata is invalid")
                decoded = ctc_greedy_decode(sample_logits, index_to_token, vocabulary.blank_index)
                counts = token_edit_counts(reference, decoded.tokens)
                records.append(
                    CTCPredictionRecord(
                        split=split,
                        sample_id=sample_id,
                        reference=reference,
                        hypothesis=decoded.tokens,
                        counts=counts,
                        path_score=decoded.path_score,
                        inference_ms=batch_ms / len(sample_ids),
                    )
                )
    corpus = corpus_wer(record.counts for record in records)
    exact = sum(record.reference == record.hypothesis for record in records)
    latencies = sorted(record.inference_ms for record in records)
    return {
        "split": split,
        "samples": len(records),
        "exact_matches": exact,
        "sequence_accuracy": exact / len(records) if records else 0.0,
        "corpus_wer": corpus,
        "inference_ms": _latency_percentiles(latencies),
    }, records


def _feature_receipt_sha256(path: Path | None) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path and path.exists() else None


def train_ctc_model(
    manifest_path: Path,
    feature_root: Path,
    vocabulary_path: Path,
    model_config_path: Path,
    training_config_path: Path,
    output_path: Path,
    feature_receipt_path: Path | None = None,
) -> dict[str, object]:
    records = read_manifest(manifest_path)
    validate_manifest(records)
    train_records = [record for record in records if record.split == "train"]
    validation_records = [record for record in records if record.split == "validation"]
    if not train_records or not validation_records:
        raise ValueError("CTC training requires official train and validation records")
    model_config = load_yaml(model_config_path)
    training_config = load_yaml(training_config_path)
    seed = int(training_config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    vocabulary = CTCVocabulary.from_file(vocabulary_path, blank_index=0)
    train_data = CTCDataset(train_records, feature_root, vocabulary)
    validation_data = CTCDataset(validation_records, feature_root, vocabulary)
    batch_size = int(training_config.get("batch_size", 8))
    train_loader = _make_loader(train_data, batch_size, shuffle=True)
    validation_loader = _make_loader(validation_data, batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_ctc_model(model_config, vocabulary.class_count).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 0.0005)),
        weight_decay=float(training_config.get("weight_decay", 0.0001)),
    )
    criterion = nn.CTCLoss(blank=vocabulary.blank_index, zero_infinity=True)
    best_wer = float("inf")
    best_state: dict[str, torch.Tensor] = {}
    history: list[dict[str, object]] = []
    patience = int(training_config.get("early_stopping_patience", 8))
    waiting = 0
    for epoch in range(1, int(training_config.get("epochs", 60)) + 1):
        model.train()
        loss_total = 0.0
        sample_total = 0
        for batch in train_loader:
            features = batch["features"]
            targets = batch["targets"]
            input_lengths = batch["input_lengths"]
            target_lengths = batch["target_lengths"]
            if not all(
                isinstance(value, torch.Tensor)
                for value in (features, targets, input_lengths, target_lengths)
            ):
                raise TypeError("CTC training batch tensors are invalid")
            optimizer.zero_grad()
            logits = model(features.to(device))
            log_probs = functional.log_softmax(logits, dim=-1).transpose(0, 1)
            loss = criterion(log_probs, targets.to(device), input_lengths, target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float(training_config.get("gradient_clip_norm", 1.0))
            )
            optimizer.step()
            loss_total += float(loss.item()) * len(features)
            sample_total += len(features)
        validation_metrics, _ = _evaluate_model(
            model, validation_loader, vocabulary, "validation", device
        )
        current_wer = float(validation_metrics["corpus_wer"]["wer"])  # type: ignore[index]
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_total / sample_total if sample_total else 0.0,
                "validation": validation_metrics,
            }
        )
        if current_wer < best_wer:
            best_wer = current_wer
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            waiting = 0
        else:
            waiting += 1
            if waiting >= patience:
                break
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_kind": "ctc_v2",
        "model_config": model_config,
        "tokens": vocabulary.tokens,
        "blank_index": vocabulary.blank_index,
        "vocabulary_sha256": hashlib.sha256(vocabulary_path.read_bytes()).hexdigest(),
        "seed": seed,
        "feature_receipt_sha256": _feature_receipt_sha256(feature_receipt_path),
        "best_validation_corpus_wer": best_wer,
        "state_dict": best_state,
    }
    torch.save(checkpoint, output_path)
    history_path = output_path.with_suffix(".history.json")
    history_path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "checkpoint": str(output_path),
        "history": str(history_path),
        "best_validation_corpus_wer": best_wer,
        "vocabulary_size": len(vocabulary.tokens),
        "device": str(device),
    }


def load_ctc_checkpoint(
    checkpoint_path: Path,
) -> tuple[nn.Module, CTCVocabulary, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if str(checkpoint.get("model_kind")) != "ctc_v2":
        raise ValueError("checkpoint is not a ctc_v2 checkpoint")
    vocabulary = CTCVocabulary(
        tokens=[str(token) for token in checkpoint["tokens"]],
        blank_index=int(checkpoint["blank_index"]),
    )
    model = build_ctc_model(checkpoint["model_config"], vocabulary.class_count)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, vocabulary, checkpoint


def evaluate_ctc_checkpoint(
    checkpoint_path: Path,
    manifest_path: Path,
    feature_root: Path,
    split: str,
) -> tuple[dict[str, object], list[CTCPredictionRecord]]:
    model, vocabulary, _ = load_ctc_checkpoint(checkpoint_path)
    records = [record for record in read_manifest(manifest_path) if record.split == split]
    if not records:
        raise ValueError(f"manifest contains no records for split: {split}")
    dataset = CTCDataset(records, feature_root, vocabulary)
    loader = _make_loader(dataset, batch_size=8, shuffle=False)
    return _evaluate_model(model, loader, vocabulary, split, torch.device("cpu"))


def evaluate_legacy_ctc_checkpoint(
    checkpoint_path: Path,
    vocabulary_path: Path,
    manifest_path: Path,
    feature_root: Path,
    split: str,
) -> tuple[dict[str, object], list[CTCPredictionRecord]]:
    vocabulary = CTCVocabulary.from_file(vocabulary_path, blank_index=0)
    vocabulary = CTCVocabulary(tokens=vocabulary.tokens, blank_index=len(vocabulary.tokens))
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_legacy_ctc_model(len(vocabulary.tokens))
    model.load_state_dict(state_dict)
    records = [record for record in read_manifest(manifest_path) if record.split == split]
    if not records:
        raise ValueError(f"manifest contains no records for split: {split}")
    dataset = CTCDataset(records, feature_root, vocabulary)
    loader = _make_loader(dataset, batch_size=8, shuffle=False)
    metrics, predictions = _evaluate_model(model, loader, vocabulary, split, torch.device("cpu"))
    metrics.update(
        {
            "model_kind": "legacy_ctc",
            "model_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "vocabulary_sha256": hashlib.sha256(vocabulary_path.read_bytes()).hexdigest(),
        }
    )
    return metrics, predictions


def evaluate_ctc_onnx(
    model_path: Path,
    vocabulary_path: Path | None,
    manifest_path: Path,
    feature_root: Path,
    split: str,
    model_kind: str,
) -> tuple[dict[str, object], list[CTCPredictionRecord]]:
    if model_kind == "legacy_ctc":
        if vocabulary_path is None:
            raise ValueError("legacy_ctc evaluation requires --vocab")
        recognizer = CTCOnnxRecognizer.legacy(model_path, vocabulary_path)
    elif model_kind == "ctc_v2":
        recognizer = CTCOnnxRecognizer.v2(model_path)
    else:
        raise ValueError(f"unsupported CTC model kind: {model_kind}")
    records = [record for record in read_manifest(manifest_path) if record.split == split]
    if not records:
        raise ValueError(f"manifest contains no records for split: {split}")
    predictions: list[CTCPredictionRecord] = []
    import time

    for record in records:
        features = np.load(feature_root / f"{record.sample_id}.npy", allow_pickle=False)
        started = time.perf_counter()
        result = recognizer.predict(features)
        elapsed = (time.perf_counter() - started) * 1000
        reference = split_ctc_tokens(record.label)
        decoded_tokens = [str(token) for token in result["tokens"]]
        predictions.append(
            CTCPredictionRecord(
                split=split,
                sample_id=record.sample_id,
                reference=reference,
                hypothesis=decoded_tokens,
                counts=token_edit_counts(reference, decoded_tokens),
                path_score=float(result["path_score"]),
                inference_ms=elapsed,
            )
        )
    metrics = corpus_wer(record.counts for record in predictions)
    exact = sum(record.reference == record.hypothesis for record in predictions)
    latencies = sorted(record.inference_ms for record in predictions)
    return {
        "split": split,
        "samples": len(predictions),
        "exact_matches": exact,
        "sequence_accuracy": exact / len(predictions) if predictions else 0.0,
        "corpus_wer": metrics,
        "inference_ms": _latency_percentiles(latencies),
        "model_kind": model_kind,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "vocabulary_sha256": (
            hashlib.sha256(vocabulary_path.read_bytes()).hexdigest() if vocabulary_path else None
        ),
    }, predictions


def write_ctc_evaluation(
    output_path: Path,
    metrics: dict[str, object],
    records: list[CTCPredictionRecord],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].as_row().keys()) if records else ["split", "sample_id"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(record.as_row() for record in records)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary_path


def export_ctc_checkpoint_to_onnx(checkpoint_path: Path, output_path: Path) -> dict[str, str]:
    model, vocabulary, checkpoint = load_ctc_checkpoint(checkpoint_path)
    input_size = int(checkpoint["model_config"]["input_size"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(1, 48, input_size)
    torch.onnx.export(
        model,
        example,
        output_path,
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    metadata_path = output_path.with_suffix(".ctc.json")
    metadata = {
        "model_kind": "ctc_v2",
        "tokens": vocabulary.tokens,
        "blank_index": vocabulary.blank_index,
        "vocabulary_sha256": str(checkpoint.get("vocabulary_sha256", "")),
        "model_config": checkpoint["model_config"],
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"model": str(output_path), "metadata": str(metadata_path)}
