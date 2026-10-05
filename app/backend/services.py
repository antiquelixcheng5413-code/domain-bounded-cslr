from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def create_recognition_service():
    """Build the best available recogniser.

    Prefer the current-best VL48 -> CTC -> Chinese-LLM chain when its resources exist;
    otherwise fall back to the legacy LSTM-onnx path so the UI still works.
    """
    # P42 landmark CTC（P23 off-by-one 修复后的模型，devWER 0.5211 / 精确 18-514）。
    # 用环境变量显式开启，避免抢掉上面那条在建链路的默认行为。
    if _as_bool(os.getenv("CSLR_USE_LANDMARK_CTC", "false")):
        from app.backend.ctc_landmark_service import create_ctc_landmark_service

        return create_ctc_landmark_service()

    from app.backend.cslr_service import VL48CtcLlmsService

    ckpt = Path(
        os.getenv(
            "CSLR_CTC_CHECKPOINT",
            str(PROJECT_ROOT / "artifacts/checkpoints/ctc-vl48-cap300.pt"),
        )
    )
    vision = os.getenv(
        "CSLR_VISION_MODEL",
        "/mnt/d/cslr-tools/models/Qwen2.5-VL-3B-Instruct",
    )
    llm = os.getenv(
        "CSLR_LLM_MODEL",
        "/mnt/d/part3_models/ms_cache/models/Qwen--Qwen2.5-1.5B-Instruct/snapshots/master",
    )
    if ckpt.exists() and Path(vision).exists() and Path(llm).exists():
        return VL48CtcLlmsService(
            vision_model=vision,
            checkpoint=str(ckpt),
            llm_model=llm,
            frames=int(os.getenv("CSLR_VL_FRAMES", "48")),
        )

    from cslr.inference.service import RecognitionService
    from cslr.semantic import IntentCatalog

    labels_config = os.getenv("CSLR_LABELS_PATH", "")
    labels_path = Path(labels_config) if labels_config else None
    configured_model = os.getenv(
        "CSLR_MODEL_PATH", str(PROJECT_ROOT / "artifacts/exports/lstm.onnx")
    )
    model_path = Path(configured_model) if configured_model else None
    catalog = IntentCatalog.from_yaml(labels_path) if labels_path else None
    return RecognitionService(
        catalog=catalog,
        model_path=model_path,
        confidence_threshold=float(os.getenv("CSLR_CONFIDENCE_THRESHOLD", "0.65")),
        demo_mode=_as_bool(os.getenv("CSLR_DEMO_MODE", "false")),
    )
