from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class IntentTemplate:
    intent: str
    gloss: str
    text_zh: str
    safety_level: str


class IntentCatalog:
    def __init__(self, templates: dict[str, IntentTemplate], fallback: IntentTemplate) -> None:
        self._templates = templates
        self._fallback = fallback
        self._by_gloss: dict[str, IntentTemplate] = {}
        for template in templates.values():
            gloss = template.gloss.upper()
            if gloss in self._by_gloss:
                raise ValueError(f"duplicate gloss in intent catalog: {gloss}")
            self._by_gloss[gloss] = template

    @classmethod
    def from_yaml(cls, path: Path) -> IntentCatalog:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}

        fallback_data = payload.get("fallback") or {}
        fallback = IntentTemplate(
            intent=str(fallback_data.get("intent", "unknown")),
            gloss=str(fallback_data.get("gloss", "UNKNOWN")),
            text_zh=str(fallback_data.get("text_zh", "未能可靠识别，请重新录制。")),
            safety_level="unknown",
        )
        templates = {}
        for intent, data in (payload.get("intents") or {}).items():
            templates[str(intent)] = IntentTemplate(
                intent=str(intent),
                gloss=str(data["gloss"]),
                text_zh=str(data["text_zh"]),
                safety_level=str(data.get("safety_level", "normal")),
            )
        if not templates:
            raise ValueError("intent catalog contains no intents")
        return cls(templates=templates, fallback=fallback)

    @property
    def intents(self) -> dict[str, IntentTemplate]:
        return dict(self._templates)

    def reconstruct(self, label: str, confidence: float, threshold: float) -> IntentTemplate:
        if confidence < threshold:
            return self._fallback
        return self._templates.get(label, self._by_gloss.get(label.upper(), self._fallback))
