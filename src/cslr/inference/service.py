from __future__ import annotations

import hashlib
import time
from pathlib import Path

from cslr.contracts import Prediction
from cslr.semantic import IntentCatalog


class RecognitionService:
    def __init__(
        self,
        catalog: IntentCatalog | None,
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
                label="unknown",
                gloss_tokens=[],
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
            return Prediction(
                status="low_quality",
                label="unknown",
                gloss_tokens=[],
                intent="unknown",
                gloss="UNKNOWN",
                text_zh="视频质量不足，无法得到可靠的 CE-CSL 识别结果。",
                confidence=0.0,
                top_k=[],
                warnings=extraction.quality.warnings,
                latency_ms={"extraction": extraction_ms, "total": self._elapsed_ms(started)},
                model_version=self.model.version,
            )

        inference_started = time.perf_counter()
        prediction = self.model.predict(extraction.features)
        inference_ms = self._elapsed_ms(inference_started)
        label = str(prediction["label"])
        tokens = [str(token) for token in prediction["gloss_tokens"]]
        confidence = float(prediction["confidence"])
        status = "ok" if confidence >= self.confidence_threshold else "low_confidence"
        warnings = [] if status == "ok" else ["prediction confidence is below threshold"]
        return Prediction(
            status=status,
            label=label,
            gloss_tokens=tokens,
            intent=label,
            gloss=label,
            text_zh=self._reconstruct_text(label, tokens, confidence),
            confidence=confidence,
            top_k=list(prediction["top_k"]),
            warnings=warnings,
            latency_ms={
                "extraction": extraction_ms,
                "inference": inference_ms,
                "total": self._elapsed_ms(started),
            },
            model_version=self.model.version,
        )

    def _demo_prediction(self, video_path: Path, started: float) -> Prediction:
        digest = hashlib.sha256(video_path.read_bytes()).digest()
        demo_tokens = ["CE-CSL-DEMO", f"TOKEN-{int.from_bytes(digest[:2], 'big') % 100:02d}"]
        label = "/".join(demo_tokens)
        confidence = 0.75
        return Prediction(
            status="demo_only",
            label=label,
            gloss_tokens=demo_tokens,
            intent=label,
            gloss=label,
            text_zh=self._reconstruct_text(label, demo_tokens, confidence),
            confidence=confidence,
            top_k=[
                {"label": token, "token": token, "confidence": confidence}
                for token in demo_tokens
            ],
            warnings=["界面演示模式：该结果不是模型识别结果，禁止用于实验报告。"],
            latency_ms={"total": self._elapsed_ms(started)},
            model_version="demo-only",
        )

    def _reconstruct_text(self, label: str, tokens: list[str], confidence: float) -> str:
        if self.catalog is not None and len(tokens) == 1:
            template = self.catalog.reconstruct(tokens[0], confidence, self.confidence_threshold)
            if template.intent != "unknown":
                return template.text_zh
        if tokens:
            return f"预测的 CE-CSL gloss/token：{' / '.join(tokens)}。"
        return f"预测标签：{label}。"

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 2)
