"""Text-generation metrics for the P2 stage (BLEU-1..4, ROUGE-L, chrF).

Implemented in-repo on purpose: no extra dependency, deterministic, and unit
tested. Tokenization is character based because CE-CSL targets are Chinese
sentences without word boundaries.

Corpus BLEU applies add-one smoothing only to orders with zero matches, so a
perfect corpus still scores exactly 1.0 while a mixed corpus does not collapse
to 0. That choice is documented in the handoff notes.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence


def _ngrams(tokens: Sequence[str], order: int) -> Counter[tuple[str, ...]]:
    if order <= 0:
        raise ValueError("order must be positive")
    return Counter(tuple(tokens[index : index + order]) for index in range(len(tokens) - order + 1))


def bleu(
    references: Sequence[Sequence[str]],
    hypotheses: Sequence[Sequence[str]],
    max_order: int = 4,
) -> dict[str, float]:
    """Corpus BLEU with add-one smoothing for empty orders."""

    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")
    if not references:
        raise ValueError("at least one reference is required")

    matches = [0] * (max_order + 1)
    totals = [0] * (max_order + 1)
    reference_length = 0
    hypothesis_length = 0
    for reference, hypothesis in zip(references, hypotheses, strict=False):
        reference_length += len(reference)
        hypothesis_length += len(hypothesis)
        for order in range(1, max_order + 1):
            reference_counts = _ngrams(reference, order)
            hypothesis_counts = _ngrams(hypothesis, order)
            totals[order] += max(sum(hypothesis_counts.values()), 0)
            matches[order] += sum(
                min(count, reference_counts[gram]) for gram, count in hypothesis_counts.items()
            )

    precisions: list[float] = []
    for order in range(1, max_order + 1):
        if totals[order] == 0:
            precisions.append(1.0 if matches[order] == 0 else 0.0)
        elif matches[order] == 0:
            precisions.append(1.0 / (totals[order] + 1.0))
        else:
            precisions.append(matches[order] / totals[order])

    if hypothesis_length == 0:
        return {f"bleu_{order}": 0.0 for order in range(1, max_order + 1)}

    if hypothesis_length > reference_length:
        brevity = 1.0
    else:
        brevity = math.exp(1.0 - reference_length / hypothesis_length)

    scores: dict[str, float] = {}
    for order in range(1, max_order + 1):
        if any(value <= 0 for value in precisions[:order]):
            scores[f"bleu_{order}"] = 0.0
            continue
        geometric = math.exp(sum(math.log(value) for value in precisions[:order]) / order)
        scores[f"bleu_{order}"] = brevity * geometric
    return scores


def rouge_l(references: Sequence[Sequence[str]], hypotheses: Sequence[Sequence[str]]) -> float:
    """Corpus-averaged ROUGE-L F1 (longest common subsequence)."""

    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")
    if not references:
        raise ValueError("at least one reference is required")
    scores: list[float] = []
    for reference, hypothesis in zip(references, hypotheses, strict=False):
        if not reference or not hypothesis:
            scores.append(0.0)
            continue
        table = [[0] * (len(hypothesis) + 1) for _ in range(len(reference) + 1)]
        for row in range(1, len(reference) + 1):
            for column in range(1, len(hypothesis) + 1):
                if reference[row - 1] == hypothesis[column - 1]:
                    table[row][column] = table[row - 1][column - 1] + 1
                else:
                    table[row][column] = max(table[row - 1][column], table[row][column - 1])
        longest = table[-1][-1]
        if longest == 0:
            scores.append(0.0)
            continue
        precision = longest / len(hypothesis)
        recall = longest / len(reference)
        scores.append(2 * precision * recall / (precision + recall))
    return sum(scores) / len(scores)


def chrf(
    references: Sequence[Sequence[str]],
    hypotheses: Sequence[Sequence[str]],
    max_order: int = 6,
    beta: float = 2.0,
) -> float:
    """Character n-gram F-score, corpus averaged."""

    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")
    if not references:
        raise ValueError("at least one reference is required")
    beta_squared = beta * beta
    scores: list[float] = []
    for reference, hypothesis in zip(references, hypotheses, strict=False):
        precisions: list[float] = []
        recalls: list[float] = []
        for order in range(1, max_order + 1):
            reference_counts = _ngrams(reference, order)
            hypothesis_counts = _ngrams(hypothesis, order)
            if not reference_counts and not hypothesis_counts:
                continue
            overlap = sum(
                min(count, reference_counts[gram]) for gram, count in hypothesis_counts.items()
            )
            precisions.append(overlap / sum(hypothesis_counts.values()) if hypothesis_counts else 0.0)
            recalls.append(overlap / sum(reference_counts.values()) if reference_counts else 0.0)
        if not precisions:
            scores.append(0.0)
            continue
        precision = sum(precisions) / len(precisions)
        recall = sum(recalls) / len(recalls)
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(
                (1 + beta_squared) * precision * recall / (beta_squared * precision + recall)
            )
    return sum(scores) / len(scores)


def exact_match(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")
    if not references:
        raise ValueError("at least one reference is required")
    return sum(
        1 for reference, hypothesis in zip(references, hypotheses, strict=False) if reference == hypothesis
    ) / len(references)


def distinct_ngrams(texts: Sequence[Sequence[str]], order: int = 1) -> dict[str, float | int]:
    """Distinct-n over a corpus of generated sequences (collapse detector)."""

    all_grams: list[tuple[str, ...]] = []
    for text in texts:
        all_grams.extend(tuple(text[index : index + order]) for index in range(len(text) - order + 1))
    if not all_grams:
        return {"order": order, "total": 0, "unique": 0, "distinct": 0.0}
    unique = len(set(all_grams))
    return {
        "order": order,
        "total": len(all_grams),
        "unique": unique,
        "distinct": unique / len(all_grams),
    }
