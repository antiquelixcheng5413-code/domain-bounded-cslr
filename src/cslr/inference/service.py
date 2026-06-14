from __future__ import annotations

import hashlib
import time
from pathlib import Path

from cslr.contracts import Prediction
from cslr.semantic import IntentCatalog


class RecognitionService:
    def __init__(
        self,
        catalog: IntentCatalog,
        model_path: Path | None,
        confidence_threshold: float = 0.65,
        demo_mode: bool = False,
    ) -> None:
        self.catalog = catalog
        self.confidence_threshold = confidence_threshold
        self.demo_mode = demo_mode
        self.model = None
        self.model_error: str | None = None
        if model_path and model_path.exists():
            try:
                from cslr.inference.onnx import OnnxRecognizer

                self.model = OnnxRecognizer(model_path)
            except (RuntimeError, FileNotFoundError, ValueError) as exc:
                self.model_error = str(exc)
        elif model_path:
            self.model_error = f"model file does not exist: {model_path}"
        else:
            self.model_error = "no model path configured"

    @property
    def ready(self) -> bool:
        return self.model is not None

    def predict_video(self, video_path: Path) -> Prediction:
        started = time.perf_counter()
        if self.model is None and not self.demo_mode:
            return Prediction(
                status="model_unavailable",
                intent="unknown",
                gloss="UNKNOWN",
                text_zh="模型尚未安装或训练，当前不能进行真实识别。",
                confidence=0.0,
                top_k=[],
                warnings=[self.model_error or "model unavailable"],
                latency_ms={"total": self._elapsed_ms(started)},
            )

        if self.demo_mode and self.model is None:
            return self._demo_prediction(video_path, started)

        extraction_started = time.perf_counter()
        from cslr.features.extractor import MediaPipeHolisticExtractor

        extraction = MediaPipeHolisticExtractor().extract(video_path)
        extraction_ms = self._elapsed_ms(extraction_started)
        if not extraction.quality.accepted:
            fallback = self.catalog.reconstruct("unknown", 0.0, self.confidence_threshold)
            return Prediction(
                status="low_quality",
                intent=fallback.intent,
                gloss=fallback.gloss,
                text_zh=fallback.text_zh,
                confidence=0.0,
                top_k=[],
                warnings=extraction.quality.warnings,
                latency_ms={"extraction": extraction_ms, "total": self._elapsed_ms(started)},
                model_version=self.model.version,
            )

        inference_started = time.perf_counter()
        intent, confidence, top_k = self.model.predict(extraction.features)
        inference_ms = self._elapsed_ms(inference_started)
        template = self.catalog.reconstruct(intent, confidence, self.confidence_threshold)
        status = "ok" if template.intent != "unknown" else "low_confidence"
        warnings = [] if status == "ok" else ["prediction confidence is below threshold"]
        return Prediction(
            status=status,
            intent=template.intent,
            gloss=template.gloss,
            text_zh=template.text_zh,
            confidence=confidence,
            top_k=top_k,
            warnings=warnings,
            latency_ms={
                "extraction": extraction_ms,
                "inference": inference_ms,
                "total": self._elapsed_ms(started),
            },
            model_version=self.model.version,
        )

    def _demo_prediction(self, video_path: Path, started: float) -> Prediction:
        intents = sorted(self.catalog.intents)
        digest = hashlib.sha256(video_path.read_bytes()).digest()
        intent = intents[int.from_bytes(digest[:2], "big") % len(intents)]
        confidence = 0.75
        template = self.catalog.reconstruct(intent, confidence, self.confidence_threshold)
        return Prediction(
            status="demo_only",
            intent=template.intent,
            gloss=template.gloss,
            text_zh=template.text_zh,
            confidence=confidence,
            top_k=[{"intent": intent, "confidence": confidence}],
            warnings=["界面演示模式：该结果不是模型识别结果，禁止用于实验报告。"],
            latency_ms={"total": self._elapsed_ms(started)},
            model_version="demo-only",
        )

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 2)
