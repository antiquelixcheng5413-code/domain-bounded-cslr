from __future__ import annotations

from pydantic import BaseModel


class RankedPrediction(BaseModel):
    intent: str
    confidence: float


class PredictionResponse(BaseModel):
    status: str
    intent: str
    gloss: str
    text_zh: str
    confidence: float
    top_k: list[RankedPrediction]
    warnings: list[str]
    latency_ms: dict[str, float]
    model_version: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    demo_mode: bool
    model_error: str | None = None
