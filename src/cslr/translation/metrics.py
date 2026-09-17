"""Dependency-free BLEU-1/2/3/4, ROUGE-L, chrF and latency metrics for Part 3.

Implemented in-repo (character-level) so evaluation is reproducible, unit
testable, and does not add a new dependency. Chinese texts are tokenized by
character.
"""

from __future__ import annotations

import math
import time
from collections import Counter

from cslr.translation.text import character_tokenize


def _tokenize(s: str) -> list[str]:
    return character_tokenize(s)


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))]


def bleu(reference: str, hypothesis: str, max_order: int = 4) -> float:
    ref = _tokenize(reference)
    hyp = _tokenize(hypothesis)
    if not ref and not hyp:
        return 1.0
    if not hyp:
        return 0.0

    ref_counts: dict[int, Counter] = {}
    for n in range(1, max_order + 1):
        ref_counts[n] = Counter(_ngrams(ref, n))

    precisions: list[float] = []
    for n in range(1, max_order + 1):
        if n > len(hyp):
            precisions.append(0.0)
            continue
        hyp_counts = Counter(_ngrams(hyp, n))
        matches = sum(
            min(hyp_counts[g], ref_counts[n].get(g, 0)) for g in hyp_counts
        )
        total = sum(hyp_counts.values())
        # zero-match orders stay 0 so a fully off-target hypothesis
        # collapses to the near-zero trail below instead of being boosted.
        precision = matches / total if total > 0 else 0.0
        precisions.append(precision)

    if any(p == 0.0 for p in precisions[: min(max_order, len(precisions))]):
        bp = math.exp(min(0.0, 1.0 - len(ref) / max(1.0, len(hyp))))
        return bp * 1e-6  # near-zero but not zero for brevity penalty

    geo = math.exp(sum(math.log(p + 1e-12) for p in precisions[:max_order]) / max_order)
    bp = math.exp(min(0.0, 1.0 - len(ref) / max(1.0, len(hyp))))
    return bp * geo


def _lcs_length(x: list[str], y: list[str]) -> int:
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l(reference: str, hypothesis: str) -> float:
    ref = _tokenize(reference)
    hyp = _tokenize(hypothesis)
    if not ref and not hyp:
        return 1.0
    if not ref or not hyp:
        return 0.0
    lcs = _lcs_length(ref, hyp)
    precision = lcs / max(1.0, len(hyp))
    recall = lcs / max(1.0, len(ref))
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def chrf(reference: str, hypothesis: str, beta: float = 2.0, ngram_order: int = 2) -> float:
    """Simplified chrF over order-1..ngram_order character n-grams."""
    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)
    if not ref_tokens and not hyp_tokens:
        return 1.0
    if not ref_tokens or not hyp_tokens:
        return 0.0

    ref_counts: Counter = Counter()
    hyp_counts: Counter = Counter()
    for n in range(1, ngram_order + 1):
        ref_counts.update(_ngrams("".join(ref_tokens), n))
        hyp_counts.update(_ngrams("".join(hyp_tokens), n))

    all_keys = set(hyp_counts) | set(ref_counts)
    matches = sum(min(hyp_counts[g], ref_counts.get(g, 0)) for g in all_keys)
    precision = matches / max(1.0, sum(hyp_counts.values()))
    recall = matches / max(1.0, sum(ref_counts.values()))
    if precision + recall == 0:
        return 0.0
    score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
    return score


class LatencyTimer:
    def __init__(self) -> None:
        self._start = 0.0

    def start(self) -> None:
        self._start = time.perf_counter()

    def stop(self) -> float:
        return (time.perf_counter() - self._start) * 1000.0  # ms