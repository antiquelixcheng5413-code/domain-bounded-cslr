from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

VALID_SPLITS = {"train", "validation", "test", "unassigned"}


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    video: Path
    label: str
    signer: str
    session: str
    split: str = "unassigned"

    def validate(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("sample_id must not be empty")
        if not self.label.strip():
            raise ValueError(f"{self.sample_id}: label must not be empty")
        if not self.signer.strip():
            raise ValueError(f"{self.sample_id}: signer must not be empty")
        if self.split not in VALID_SPLITS:
            raise ValueError(
                f"{self.sample_id}: split must be one of {sorted(VALID_SPLITS)}, got {self.split}"
            )
        if self.video.is_absolute():
            raise ValueError(f"{self.sample_id}: video path must be relative to dataset root")
        if ".." in self.video.parts:
            raise ValueError(f"{self.sample_id}: video path must not escape dataset root")


@dataclass(frozen=True)
class QualityReport:
    total_frames: int
    valid_frames: int
    valid_ratio: float
    accepted: bool
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Prediction:
    status: str
    label: str
    gloss_tokens: list[str]
    intent: str
    gloss: str
    text_zh: str
    confidence: float
    top_k: list[dict[str, object]]
    warnings: list[str]
    latency_ms: dict[str, float]
    model_version: str | None = None
