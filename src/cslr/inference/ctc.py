from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from cslr.data.ctc import CTCVocabulary
from cslr.evaluation.ctc import ctc_greedy_decode


class CTCOnnxRecognizer:
    """ONNX Runtime CTC recognizer shared by legacy evaluation and the web service."""

    def __init__(
        self,
        model_path: Path,
        vocabulary: CTCVocabulary,
        model_kind: str,
        model_version: str | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is not installed") from exc
        self.vocabulary = vocabulary
        self.model_kind = model_kind
        self.version = model_version or model_path.stem
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        output_shape = self.session.get_outputs()[0].shape
        output_classes = output_shape[-1] if output_shape else None
        if isinstance(output_classes, int) and output_classes != vocabulary.class_count:
            raise ValueError(
                "CTC vocabulary/model mismatch: "
                f"vocabulary expects {vocabulary.class_count} classes, "
                f"model outputs {output_classes}"
            )

    @classmethod
    def legacy(cls, model_path: Path, vocabulary_path: Path) -> CTCOnnxRecognizer:
        vocabulary = CTCVocabulary.from_file(vocabulary_path, blank_index=0)
        # The delivered legacy model used the final vocabulary index as blank.
        vocabulary = CTCVocabulary(tokens=vocabulary.tokens, blank_index=len(vocabulary.tokens))
        return cls(model_path, vocabulary, model_kind="legacy_ctc", model_version="legacy_ctc")

    @classmethod
    def v2(cls, model_path: Path) -> CTCOnnxRecognizer:
        metadata_path = model_path.with_suffix(".ctc.json")
        if not metadata_path.exists():
            raise FileNotFoundError(f"CTC metadata is missing: {metadata_path.name}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("model_kind") != "ctc_v2":
            raise ValueError("CTC metadata must declare model_kind=ctc_v2")
        vocabulary = CTCVocabulary(
            tokens=[str(token) for token in metadata["tokens"]],
            blank_index=int(metadata["blank_index"]),
        )
        return cls(
            model_path,
            vocabulary,
            model_kind="ctc_v2",
            model_version=str(metadata_path.stem),
        )

    def predict(self, features: np.ndarray) -> dict[str, Any]:
        if features.ndim != 2:
            raise ValueError(f"expected [frames, features], got {features.shape}")
        logits = self.session.run(
            None, {self.input_name: features[np.newaxis, :, :].astype(np.float32)}
        )[0][0]
        decoded = ctc_greedy_decode(
            logits,
            self.vocabulary.index_to_token,
            self.vocabulary.blank_index,
        )
        return {
            "tokens": decoded.tokens,
            "path_score": decoded.path_score,
            "logits": logits,
            "model_kind": self.model_kind,
            "model_version": self.version,
        }
