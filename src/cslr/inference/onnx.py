from __future__ import annotations

import json
from pathlib import Path

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
        self.version = str(metadata.get("version", model_path.stem))
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, features: np.ndarray, top_k: int = 3) -> tuple[str, float, list[dict]]:
        logits = self.session.run(None, {self.input_name: features[None].astype(np.float32)})[0][0]
        shifted = logits - np.max(logits)
        probabilities = np.exp(shifted) / np.exp(shifted).sum()
        indices = np.argsort(probabilities)[::-1][:top_k]
        ranked = [
            {"intent": self.labels[index], "confidence": float(probabilities[index])}
            for index in indices
        ]
        best = int(indices[0])
        return self.labels[best], float(probabilities[best]), ranked
