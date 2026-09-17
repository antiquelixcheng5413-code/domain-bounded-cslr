"""Chinese text normalization and character-level tokenization for Part 3.

Normalization only: collapses whitespace and removes zero-width characters.
It never rewrites punctuation or word choices, and always keeps the raw text.
"""

from __future__ import annotations

from dataclasses import dataclass

import unicodedata

_ZERO_WIDTH = {ord(c) for c in "\u200b\u200c\u200d\u2060\ufeff"}


@dataclass(frozen=True)
class NormalizedText:
    raw: str
    normalized: str
    changed: bool


class TranslationTextNormalizer:
    """Deterministic, lossless-as-possible Chinese text normalizer."""

    def __call__(self, text: str) -> NormalizedText:
        return self.normalize(text)

    @staticmethod
    def normalize(text: str) -> NormalizedText:
        raw = text or ""
        stripped_zero_width = "".join(ch for ch in raw if ord(ch) not in _ZERO_WIDTH)
        collapsed = " ".join(stripped_zero_width.split())
        return NormalizedText(raw=raw, normalized=collapsed, changed=collapsed != raw)


def character_tokenize(text: str) -> list[str]:
    """Character-level tokenization of a normalized Chinese string.

    ASCII runs are kept as compact tokens; CJK characters are separated.
    """
    tokens: list[str] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if current:
            tokens.append("".join(current))
            current = []

    for char in text:
        if char.isspace():
            flush()
            continue
        if char.isascii():
            current.append(char)
        else:
            flush()
            tokens.append(char)
    flush()
    return tokens


def build_vocab(sentences: list[str], special_tokens: tuple[str, ...] = ("<pad>", "<bos>", "<eos>", "<unk>")) -> dict[str, int]:
    """Build a character-level vocabulary from normalized sentences.

    Order is stable: special tokens first, then the remaining characters in
    first-seen order.
    """
    vocab: dict[str, int] = {}
    for token in special_tokens:
        vocab[token] = len(vocab)
    for sentence in sentences:
        for token in character_tokenize(sentence):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab