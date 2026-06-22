from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from cslr.data.ctc import split_ctc_tokens


@dataclass(frozen=True)
class SemanticReference:
    status: str
    text_zh: str
    sample_id: str | None = None


class ExactSemanticResolver:
    """Resolve a predicted Gloss sequence only when CE-CSL supplies one sentence reference."""

    def __init__(self, references: dict[str, tuple[str, str] | None], available: bool) -> None:
        self.references = references
        self.available = available

    @classmethod
    def from_ce_csl(cls, data_root: Path | None) -> ExactSemanticResolver:
        if data_root is None:
            return cls({}, available=False)
        label_root = data_root / "label"
        paths = [label_root / name for name in ("train.csv", "dev.csv", "test.csv")]
        if not all(path.exists() for path in paths):
            return cls({}, available=False)
        candidates: dict[str, set[tuple[str, str]]] = {}
        for path in paths:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    tokens = split_ctc_tokens((row.get("Gloss") or "").strip())
                    sentence = (row.get("Chinese Sentences") or "").strip()
                    sample_id = (row.get("Number") or "").strip()
                    if not tokens or not sentence or not sample_id:
                        continue
                    key = "/".join(tokens)
                    candidates.setdefault(key, set()).add((sample_id, sentence))
        references = {
            key: next(iter(values)) if len({sentence for _, sentence in values}) == 1 else None
            for key, values in candidates.items()
        }
        return cls(references, available=True)

    def resolve(self, tokens: list[str]) -> SemanticReference:
        if not self.available:
            return SemanticReference("reference_unavailable", "无可靠中文重构。")
        key = "/".join(tokens)
        if not key or key not in self.references:
            return SemanticReference("no_exact_reference", "无可靠中文重构。")
        reference = self.references[key]
        if reference is None:
            return SemanticReference("ambiguous_reference", "无可靠中文重构。")
        sample_id, sentence = reference
        return SemanticReference("exact_reference", sentence, sample_id)
