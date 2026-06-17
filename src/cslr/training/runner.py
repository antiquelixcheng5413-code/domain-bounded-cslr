from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from cslr.config import load_yaml
from cslr.data.gloss import build_gloss_vocabulary
from cslr.data.manifest import read_manifest, validate_manifest
from cslr.evaluation.metrics import classification_metrics, multilabel_metrics
from cslr.models import build_model
from cslr.training.dataset import GlossTokenDataset, LandmarkDataset


def train_model(
    manifest_path: Path,
    feature_root: Path,
    model_config_path: Path,
    training_config_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    records = read_manifest(manifest_path)
    validate_manifest(records)
    train_records = [record for record in records if record.split == "train"]
    validation_records = [record for record in records if record.split == "validation"]
    if not train_records or not validation_records:
        raise ValueError("manifest must contain train and validation samples")

    model_config = load_yaml(model_config_path)
    training_config = load_yaml(training_config_path)
    seed = int(training_config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    task = str(training_config.get("task", "single_label_classification"))
    multilabel_task = task in {"gloss_token_multilabel", "multi_label", "multilabel"}
    if multilabel_task:
        min_frequency = int(training_config.get("min_token_frequency", 2))
        max_tokens_config = training_config.get("max_tokens")
        max_tokens = int(max_tokens_config) if max_tokens_config is not None else None
        labels, token_counts = build_gloss_vocabulary(
            train_records,
            min_frequency=min_frequency,
            max_tokens=max_tokens,
        )
        if not labels:
            raise ValueError("gloss token vocabulary is empty; lower min_token_frequency")
        label_to_index = {label: index for index, label in enumerate(labels)}
        train_data = GlossTokenDataset(train_records, feature_root, label_to_index)
        validation_data = GlossTokenDataset(validation_records, feature_root, label_to_index)
        criterion: nn.Module = nn.BCEWithLogitsLoss()
    else:
        token_counts = {}
        labels = sorted({record.label for record in records})
        label_to_index = {label: index for index, label in enumerate(labels)}
        train_data = LandmarkDataset(train_records, feature_root, label_to_index)
        validation_data = LandmarkDataset(validation_records, feature_root, label_to_index)
        criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(
        train_data,
        batch_size=int(training_config.get("batch_size", 32)),
        shuffle=True,
        num_workers=int(training_config.get("num_workers", 0)),
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=int(training_config.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(training_config.get("num_workers", 0)),
    )

    model = build_model(model_config, len(labels))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 0.001)),
        weight_decay=float(training_config.get("weight_decay", 0.0001)),
    )
    epochs = int(training_config.get("epochs", 50))
    prediction_threshold = float(training_config.get("prediction_threshold", 0.5))
    best_macro_f1 = -1.0
    best_state: dict[str, torch.Tensor] = {}
    history: list[dict[str, float]] = []
    patience = int(training_config.get("early_stopping_patience", 8))
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for features, targets in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * len(targets)

        model.eval()
        if multilabel_task:
            expected_sets: list[list[str]] = []
            predicted_sets: list[list[str]] = []
            with torch.no_grad():
                for features, targets in validation_loader:
                    probabilities = torch.sigmoid(model(features))
                    predictions = probabilities >= prediction_threshold
                    for expected_row, predicted_row in zip(
                        targets.bool().tolist(), predictions.tolist(), strict=False
                    ):
                        expected_sets.append(
                            [labels[index] for index, present in enumerate(expected_row) if present]
                        )
                        predicted_sets.append(
                            [
                                labels[index]
                                for index, present in enumerate(predicted_row)
                                if present
                            ]
                        )
            metrics = multilabel_metrics(expected_sets, predicted_sets, labels)
            validation_accuracy = float(metrics["subset_accuracy"])
            validation_micro_f1 = float(metrics["micro_f1"])
        else:
            expected_labels: list[str] = []
            predicted_labels: list[str] = []
            with torch.no_grad():
                for features, targets in validation_loader:
                    predictions = model(features).argmax(dim=1)
                    expected_labels.extend(labels[index] for index in targets.tolist())
                    predicted_labels.extend(labels[index] for index in predictions.tolist())
            metrics = classification_metrics(expected_labels, predicted_labels)
            validation_accuracy = float(metrics["accuracy"])
            validation_micro_f1 = validation_accuracy
        validation_macro_f1 = float(metrics["macro_f1"])
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss / len(train_data),
                "validation_accuracy": validation_accuracy,
                "validation_micro_f1": validation_micro_f1,
                "validation_macro_f1": validation_macro_f1,
            }
        )
        if validation_macro_f1 > best_macro_f1:
            best_macro_f1 = validation_macro_f1
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": model_config,
            "task": "gloss_token_multilabel" if multilabel_task else "single_label_classification",
            "labels": labels,
            "label_counts": token_counts,
            "prediction_threshold": prediction_threshold,
            "state_dict": best_state,
            "validation_macro_f1": best_macro_f1,
        },
        output_path,
    )
    history_path = output_path.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {
        "task": "gloss_token_multilabel" if multilabel_task else "single_label_classification",
        "labels": labels,
        "label_count": len(labels),
        "best_validation_macro_f1": best_macro_f1,
    }
