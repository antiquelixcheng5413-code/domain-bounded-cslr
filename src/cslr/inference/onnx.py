from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class OnnxRecognizer:
    def __init__(self, model_path: Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is not installed") from exc

        metadata_path = model_path.with_suffix(".labels.json")
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"model labels are missing: expected {metadata_path.name} beside the ONNX model"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.labels: list[str] = [str(label) for label in metadata["labels"]]
        self.task = str(metadata.get("task", "single_label_classification"))
        self.prediction_threshold = float(metadata.get("prediction_threshold", 0.5))
        self.version = str(metadata.get("version", model_path.stem))
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, features: np.ndarray, top_k: int = 5) -> dict[str, Any]:
        logits = self.session.run(None, {self.input_name: features[None].astype(np.float32)})[0][0]
        if self.task == "gloss_token_multilabel":
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            indices = np.argsort(probabilities)[::-1][:top_k]
            ranked = [
                {
                    "label": self.labels[index],
                    "token": self.labels[index],
                    "confidence": float(probabilities[index]),
                }
                for index in indices
            ]
            predicted_indices = [
                int(index)
                for index in np.argsort(probabilities)[::-1]
                if probabilities[index] >= self.prediction_threshold
            ]
            if not predicted_indices and len(indices):
                predicted_indices = [int(indices[0])]
            tokens = [self.labels[index] for index in predicted_indices]
            confidence = float(
                max((probabilities[index] for index in predicted_indices), default=0.0)
            )
            return {
                "task": self.task,
                "label": "/".join(tokens) if tokens else "UNKNOWN",
                "gloss_tokens": tokens,
                "confidence": confidence,
                "top_k": ranked,
            }

        shifted = logits - np.max(logits)
        probabilities = np.exp(shifted) / np.exp(shifted).sum()
        indices = np.argsort(probabilities)[::-1][:top_k]
        ranked = [
            {
                "label": self.labels[index],
                "intent": self.labels[index],
                "confidence": float(probabilities[index]),
            }
            for index in indices
        ]
        best = int(indices[0])
        return {
            "task": self.task,
            "label": self.labels[best],
            "gloss_tokens": [self.labels[best]],
            "confidence": float(probabilities[best]),
            "top_k": ranked,
        }
