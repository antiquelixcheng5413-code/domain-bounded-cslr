from __future__ import annotations

import os
from pathlib import Path

from cslr.inference.service import RecognitionService

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def create_recognition_service() -> RecognitionService:
    configured_model = os.getenv(
        "CSLR_MODEL_PATH", str(PROJECT_ROOT / "artifacts/exports/lstm.onnx")
    )
    model_path = Path(configured_model) if configured_model else None
    configured_data_root = os.getenv("CSLR_SEMANTIC_DATA_ROOT", "")
    semantic_data_root = Path(configured_data_root) if configured_data_root else None
    ctc_vocabulary = os.getenv("CSLR_CTC_VOCAB_PATH", "")
    return RecognitionService(
        model_path=model_path,
        confidence_threshold=float(os.getenv("CSLR_CONFIDENCE_THRESHOLD", "0.65")),
        demo_mode=_as_bool(os.getenv("CSLR_DEMO_MODE", "false")),
        model_kind=os.getenv("CSLR_MODEL_KIND", "multilabel"),
        ctc_vocabulary_path=Path(ctc_vocabulary) if ctc_vocabulary else None,
        semantic_data_root=semantic_data_root,
        allow_legacy_ctc=_as_bool(os.getenv("CSLR_ENABLE_LEGACY_CTC", "false")),
    )
