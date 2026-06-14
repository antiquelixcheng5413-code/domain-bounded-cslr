from __future__ import annotations

import os
from pathlib import Path

from cslr.inference.service import RecognitionService
from cslr.semantic import IntentCatalog

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def create_recognition_service() -> RecognitionService:
    labels_path = Path(
        os.getenv("CSLR_LABELS_PATH", str(PROJECT_ROOT / "configs/hospital_intents.yaml"))
    )
    configured_model = os.getenv(
        "CSLR_MODEL_PATH", str(PROJECT_ROOT / "artifacts/exports/lstm.onnx")
    )
    model_path = Path(configured_model) if configured_model else None
    return RecognitionService(
        catalog=IntentCatalog.from_yaml(labels_path),
        model_path=model_path,
        confidence_threshold=float(os.getenv("CSLR_CONFIDENCE_THRESHOLD", "0.65")),
        demo_mode=_as_bool(os.getenv("CSLR_DEMO_MODE", "false")),
    )
