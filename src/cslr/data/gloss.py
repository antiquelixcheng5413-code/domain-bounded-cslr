from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from cslr.contracts import SampleRecord

PUNCTUATION_TOKENS = {
    "",
    ".",
    ",",
    "?",
    "!",
    ";",
    ":",
    "。",
    "，",
    "、",
    "？",
    "！",
    "；",
    "：",
}


def split_gloss_tokens(gloss: str) -> list[str]:
    """Split a CE-CSL gloss string into stable token labels."""
    tokens: list[str] = []
    for token in gloss.replace("\u3000", " ").split("/"):
        cleaned = token.strip()
        if cleaned not in PUNCTUATION_TOKENS:
            tokens.append(cleaned)
    return tokens


def count_gloss_tokens(records: Iterable[SampleRecord]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(set(split_gloss_tokens(record.label)))
    return counts


def build_gloss_vocabulary(
    records: Iterable[SampleRecord],
    min_frequency: int = 2,
    max_tokens: int | None = None,
) -> tuple[list[str], dict[str, int]]:
    if min_frequency < 1:
        raise ValueError("min_frequency must be at least 1")

    counts = count_gloss_tokens(records)
    candidates = [
        token for token, count in counts.items() if count >= min_frequency
    ]
    tokens = sorted(candidates, key=lambda token: (-counts[token], token))
    if max_tokens is not None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        tokens = tokens[:max_tokens]
    return tokens, {token: counts[token] for token in tokens}


def encode_gloss_tokens(gloss: str, token_to_index: dict[str, int]) -> list[float]:
    encoded = [0.0] * len(token_to_index)
    for token in set(split_gloss_tokens(gloss)):
        index = token_to_index.get(token)
        if index is not None:
            encoded[index] = 1.0
    return encoded
