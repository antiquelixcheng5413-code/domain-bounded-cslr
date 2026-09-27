"""Ordered CE-CSL gloss sequences for CTC training.

The Part 1/2 vocabulary in :mod:`cslr.data.gloss` treats a gloss string as an
*unordered set* of tokens (multi-hot encoding). CTC needs the opposite: an
ordered, variable-length sequence plus a reproducible vocabulary. This module
adds that ordered view without modifying the Part 1/2 behaviour.

Cleaning rules are configuration driven because the raw CE-CSL ``Gloss`` field
mixes three different things (measured on the official train split):

- variant numbering, e.g. ``考试1`` / ``不行2`` -- the same Chinese word signed in
  a different way. 1469 token occurrences / 372 distinct strings.
- pragmatic annotations, e.g. ``带1（我）`` / ``英国{英国词汇第1个手势动作}``.
- legitimate number+measure words, e.g. ``2个`` / ``9折`` / ``2.2亿`` / ``10月3号``.

Stripping variant numbering may erase a real sign distinction, so it stays a
configuration flag and every cleaning step records what it changed.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

UNKNOWN_TOKEN = "<unk>"
PUNCTUATION_TOKENS = frozenset(
    {"", ".", ",", "?", "!", ";", ":", "。", "，", "、", "？", "！", "；", "：", "…", "~"}
)

# ``考试1`` -> ("考试", "1"); only applies when the token ends in digits.
_TRAILING_VARIANT = re.compile(r"^(?P<base>.*?)(?P<variant>\d+)$")
# Bracketed annotations: （我）(我) {注释} [注释]
_ANNOTATION = re.compile(r"[（(][^（()）]*[)）]|\{[^{}]*\}|\[[^\[\]]*\]")
# Number + measure word / currency / clock, e.g. 2个 9折 2.2亿 10月3号 50元
_NUMBER_WITH_TAIL = re.compile(r"^\d+(\.\d+)?[^\d]*$")


@dataclass(frozen=True)
class GlossSequenceConfig:
    """How raw ``Gloss`` strings are turned into ordered token sequences."""

    keep_punctuation: bool = True
    keep_numeric_tokens: bool = True
    strip_variant_numbering: bool = True
    strip_annotations: bool = True
    # "token" predicts whole gloss tokens, "char" predicts individual characters. The character
    # view exists because CE-CSL token labels are extremely sparse (73% of tokens occur once or
    # twice on a small train split), while characters are ~4x denser and give CTC longer targets
    # to align against. The cost is that it no longer emits gloss tokens directly.
    target_unit: str = "token"

    def as_dict(self) -> dict[str, object]:
        return {
            "keep_punctuation": self.keep_punctuation,
            "keep_numeric_tokens": self.keep_numeric_tokens,
            "strip_variant_numbering": self.strip_variant_numbering,
            "strip_annotations": self.strip_annotations,
            "target_unit": self.target_unit,
        }

    def validate(self) -> None:
        if self.target_unit not in {"token", "char"}:
            raise ValueError(f"unsupported target_unit: {self.target_unit}")


@dataclass
class CleaningStats:
    """Counters proving what the cleaning rules actually did."""

    raw_tokens: int = 0
    kept_tokens: int = 0
    dropped_punctuation: int = 0
    dropped_numeric: int = 0
    dropped_empty: int = 0
    annotation_stripped: int = 0
    variant_stripped: int = 0
    samples: int = 0
    samples_changed: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "raw_tokens": self.raw_tokens,
            "kept_tokens": self.kept_tokens,
            "dropped_punctuation": self.dropped_punctuation,
            "dropped_numeric": self.dropped_numeric,
            "dropped_empty": self.dropped_empty,
            "annotation_stripped": self.annotation_stripped,
            "variant_stripped": self.variant_stripped,
            "samples": self.samples,
            "samples_changed": self.samples_changed,
        }


def split_raw_gloss(gloss: str) -> list[str]:
    """Split a raw ``Gloss`` string on ``/`` and strip each part."""

    return [token.strip() for token in gloss.replace("\u3000", " ").split("/")]


def _is_punctuation(token: str) -> bool:
    return token in PUNCTUATION_TOKENS


def _is_numeric(token: str) -> bool:
    return bool(token) and all(character.isdigit() for character in token)


def clean_token(
    token: str, config: GlossSequenceConfig, stats: CleaningStats | None = None
) -> str | None:
    """Apply the annotation / variant / punctuation rules to one token.

    Returns the canonical token, or ``None`` when the token must be dropped.
    ``stats`` is updated in place when provided.
    """

    cleaned = token
    if config.strip_annotations:
        without_annotations = _ANNOTATION.sub("", cleaned)
        if without_annotations != cleaned:
            if stats is not None:
                stats.annotation_stripped += 1
            cleaned = without_annotations.strip()

    if not cleaned:
        if stats is not None:
            stats.dropped_empty += 1
        return None

    if config.strip_variant_numbering and not _is_numeric(cleaned):
        match = _TRAILING_VARIANT.match(cleaned)
        if match is not None:
            base = match.group("base").strip()
            if base and not _is_numeric(base):
                if stats is not None:
                    stats.variant_stripped += 1
                cleaned = base

    if _is_punctuation(cleaned):
        if config.keep_punctuation:
            return cleaned
        if stats is not None:
            stats.dropped_punctuation += 1
        return None

    if _is_numeric(cleaned) and not config.keep_numeric_tokens:
        if stats is not None:
            stats.dropped_numeric += 1
        return None

    return cleaned


def split_gloss_sequence(
    gloss: str,
    config: GlossSequenceConfig | None = None,
    stats: CleaningStats | None = None,
) -> list[str]:
    """Turn one raw ``Gloss`` string into an ordered canonical token sequence."""

    resolved = config or GlossSequenceConfig()
    tokens: list[str] = []
    raw_tokens = split_raw_gloss(gloss)
    if stats is not None:
        stats.samples += 1
        stats.raw_tokens += len([token for token in raw_tokens if token])
    for raw in raw_tokens:
        if not raw:
            if stats is not None:
                stats.dropped_empty += 1
            continue
        cleaned = clean_token(raw, resolved, stats)
        if cleaned is not None:
            tokens.append(cleaned)
    if stats is not None:
        stats.kept_tokens += len(tokens)
        if tokens != [token for token in raw_tokens if token]:
            stats.samples_changed += 1
    return tokens


def token_is_numeric(token: str) -> bool:
    """True for pure-digit tokens such as ``2`` and ``20``."""

    return _is_numeric(token)


def token_is_number_with_tail(token: str) -> bool:
    """True for number+measure tokens such as ``2个``, ``9折``, ``2.2亿``."""

    return bool(_NUMBER_WITH_TAIL.match(token)) and not _is_numeric(token)


@dataclass(frozen=True)
class GlossVocabulary:
    """Ordered-token vocabulary. Index 0 is reserved for ``<unk>``."""

    tokens: tuple[str, ...]
    counts: dict[str, int] = field(default_factory=dict)
    config: GlossSequenceConfig = field(default_factory=GlossSequenceConfig)

    @property
    def unknown_index(self) -> int:
        return 0

    @property
    def size(self) -> int:
        return len(self.tokens)

    def __contains__(self, token: object) -> bool:
        return token in self.tokens

    def index_of(self, token: str) -> int:
        try:
            return self.tokens.index(token)
        except ValueError:
            return self.unknown_index

    def units(self, gloss: str) -> list[str]:
        """Ordered prediction units for a raw gloss string (tokens or characters)."""

        tokens = split_gloss_sequence(gloss, self.config)
        if self.config.target_unit == "char":
            return [character for token in tokens for character in token]
        return tokens

    def encode(self, gloss: str) -> list[int]:
        """Encode a raw gloss string into ordered unit ids."""

        return [self.index_of(unit) for unit in self.units(gloss)]

    def encode_tokens(self, tokens: Sequence[str]) -> list[int]:
        return [self.index_of(token) for token in tokens]

    def decode(self, indices: Iterable[int], drop_unknown: bool = False) -> list[str]:
        tokens: list[str] = []
        for index in indices:
            if index < 0 or index >= self.size:
                tokens.append(UNKNOWN_TOKEN)
                continue
            token = self.tokens[index]
            if drop_unknown and token == UNKNOWN_TOKEN:
                continue
            tokens.append(token)
        return tokens

    def as_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for index, token in enumerate(self.tokens):
            rows.append(
                {
                    "index": index,
                    "token": token,
                    "frequency": self.counts.get(token, 0),
                }
            )
        return rows

    def to_json(self) -> dict[str, object]:
        return {
            "tokens": list(self.tokens),
            "counts": dict(self.counts),
            "config": self.config.as_dict(),
        }

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> GlossVocabulary:
        raw_config = payload.get("config") or {}
        config = GlossSequenceConfig(
            keep_punctuation=bool(raw_config.get("keep_punctuation", True)),
            keep_numeric_tokens=bool(raw_config.get("keep_numeric_tokens", True)),
            strip_variant_numbering=bool(raw_config.get("strip_variant_numbering", True)),
            strip_annotations=bool(raw_config.get("strip_annotations", True)),
            target_unit=str(raw_config.get("target_unit", "token")),
        )
        config.validate()
        return cls(
            tokens=tuple(str(token) for token in payload["tokens"]),
            counts={str(key): int(value) for key, value in dict(payload.get("counts") or {}).items()},
            config=config,
        )


def token_counts(
    glosses: Iterable[str], config: GlossSequenceConfig | None = None
) -> Counter[str]:
    """Count ordered prediction-unit occurrences across raw gloss strings.

    With ``target_unit="char"`` the units are characters, otherwise whole gloss tokens.
    """

    resolved = config or GlossSequenceConfig()
    counts: Counter[str] = Counter()
    for gloss in glosses:
        tokens = split_gloss_sequence(gloss, resolved)
        if resolved.target_unit == "char":
            counts.update(character for token in tokens for character in token)
        else:
            counts.update(tokens)
    return counts


def build_ordered_vocabulary(
    glosses: Iterable[str],
    min_frequency: int = 2,
    max_tokens: int | None = None,
    config: GlossSequenceConfig | None = None,
) -> tuple[GlossVocabulary, Counter[str]]:
    """Build a reproducible ordered vocabulary.

    Units are sorted by descending frequency, then by unit string, so the result is
    deterministic. ``<unk>`` always occupies index 0.
    """

    if min_frequency < 1:
        raise ValueError("min_frequency must be at least 1")
    resolved = config or GlossSequenceConfig()
    resolved.validate()
    counts = token_counts(glosses, resolved)
    candidates = [token for token, count in counts.items() if count >= min_frequency]
    candidates.sort(key=lambda token: (-counts[token], token))
    if max_tokens is not None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        candidates = candidates[:max_tokens]
    tokens = (UNKNOWN_TOKEN, *candidates)
    vocabulary = GlossVocabulary(
        tokens=tokens,
        counts={token: counts[token] for token in candidates},
        config=resolved,
    )
    return vocabulary, counts


def coverage_report(
    glosses: Sequence[str],
    vocabulary: GlossVocabulary,
    counts: Counter[str] | None = None,
) -> dict[str, object]:
    """Report OOV pressure and sequence-length statistics for a vocabulary."""

    total = 0
    in_vocabulary = 0
    unknown = 0
    lengths: list[int] = []
    empty_sequences = 0
    numeric_tokens = 0
    number_tail_tokens = 0
    punctuation_tokens = 0
    unknown_counter: Counter[str] = Counter()
    for gloss in glosses:
        tokens = vocabulary.units(gloss)
        lengths.append(len(tokens))
        if not tokens:
            empty_sequences += 1
        for token in tokens:
            total += 1
            if token in vocabulary:
                in_vocabulary += 1
            else:
                unknown += 1
                unknown_counter[token] += 1
            if token_is_numeric(token):
                numeric_tokens += 1
            elif token_is_number_with_tail(token):
                number_tail_tokens += 1
            if _is_punctuation(token):
                punctuation_tokens += 1
    ordered = sorted(lengths)
    return {
        "samples": len(lengths),
        "token_occurrences": total,
        "in_vocabulary": in_vocabulary,
        "unknown": unknown,
        "oov_rate": (unknown / total) if total else 0.0,
        "vocabulary_size": vocabulary.size,
        "empty_sequences": empty_sequences,
        "numeric_tokens": numeric_tokens,
        "number_with_tail_tokens": number_tail_tokens,
        "punctuation_tokens": punctuation_tokens,
        "sequence_length": {
            "min": ordered[0] if ordered else 0,
            "p25": _percentile(ordered, 0.25),
            "median": _percentile(ordered, 0.50),
            "p75": _percentile(ordered, 0.75),
            "p90": _percentile(ordered, 0.90),
            "max": ordered[-1] if ordered else 0,
            "mean": (sum(ordered) / len(ordered)) if ordered else 0.0,
        },
        "top_unknown": unknown_counter.most_common(20),
        "source_counts": dict(counts) if counts is not None else {},
    }


def _percentile(ordered: Sequence[int], fraction: float) -> int:
    if not ordered:
        return 0
    position = int(math.floor(len(ordered) * fraction))
    position = min(max(position, 0), len(ordered) - 1)
    return int(ordered[position])
