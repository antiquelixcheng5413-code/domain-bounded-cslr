"""Edit-distance metrics for ordered gloss sequences.

These are deliberately dependency free (no ``jiwer``/``editdistance``): the
repository keeps its evaluation code reproducible and unit tested in-repo.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Levenshtein distance with substitutions costing 1."""

    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)
    previous = list(range(len(hypothesis) + 1))
    for row, reference_token in enumerate(reference, start=1):
        current = [row]
        for column, hypothesis_token in enumerate(hypothesis, start=1):
            cost = 0 if reference_token == hypothesis_token else 1
            current.append(
                min(
                    previous[column] + 1,  # deletion
                    current[column - 1] + 1,  # insertion
                    previous[column - 1] + cost,  # substitution / match
                )
            )
        previous = current
    return previous[-1]


def error_rate(references: Sequence[Sequence[str]], hypotheses: Sequence[Sequence[str]]) -> float:
    """Total edit distance divided by total reference length."""

    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")
    if not references:
        raise ValueError("at least one reference is required")
    total_errors = 0
    total_length = 0
    for reference, hypothesis in zip(references, hypotheses, strict=False):
        total_errors += edit_distance(reference, hypothesis)
        total_length += max(len(reference), 1)
    return total_errors / total_length if total_length else 0.0


def sequence_exact_match(
    references: Sequence[Sequence[str]], hypotheses: Sequence[Sequence[str]]
) -> float:
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")
    if not references:
        raise ValueError("at least one reference is required")
    matches = sum(
        1
        for reference, hypothesis in zip(references, hypotheses, strict=False)
        if list(reference) == list(hypothesis)
    )
    return matches / len(references)


def character_error_rate(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """Edit distance over individual characters (reference strings)."""

    ref_chars = [list(reference) for reference in references]
    hyp_chars = [list(hypothesis) for hypothesis in hypotheses]
    return error_rate(ref_chars, hyp_chars)


def distinct_ratio(sequences: Sequence[Sequence[str]]) -> dict[str, float | int]:
    """Collapse detector: how many distinct sequences were produced."""

    total = len(sequences)
    if total == 0:
        return {"count": 0, "unique": 0, "distinct_ratio": 0.0, "most_common_count": 0}
    as_tuples = [tuple(sequence) for sequence in sequences]
    counter = Counter(as_tuples)
    most_common_count = counter.most_common(1)[0][1]
    return {
        "count": total,
        "unique": len(counter),
        "distinct_ratio": len(counter) / total,
        "most_common_count": most_common_count,
        "most_common": list(counter.most_common(1)[0][0]),
    }


def gloss_metrics(
    references: Sequence[Sequence[str]],
    hypotheses: Sequence[Sequence[str]],
    vocabulary_size: int | None = None,
    blank_steps: int = 0,
    total_steps: int = 0,
) -> dict[str, Any]:
    """One dictionary with every number the plan requires for the P1 stage."""

    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")
    if not references:
        raise ValueError("at least one reference is required")

    distances = [
        edit_distance(reference, hypothesis)
        for reference, hypothesis in zip(references, hypotheses, strict=False)
    ]
    used_tokens = Counter(
        token for hypothesis in hypotheses for token in hypothesis
    )
    empty_hypotheses = sum(1 for hypothesis in hypotheses if len(hypothesis) == 0)
    empty_references = sum(1 for reference in references if len(reference) == 0)
    metrics: dict[str, Any] = {
        "samples": len(references),
        "wer": error_rate(references, hypotheses),
        "cer": character_error_rate(
            ["".join(reference) for reference in references],
            ["".join(hypothesis) for hypothesis in hypotheses],
        ),
        "sequence_exact_match": sequence_exact_match(references, hypotheses),
        "total_edit_distance": sum(distances),
        "mean_edit_distance": sum(distances) / len(distances),
        "empty_hypotheses": empty_hypotheses,
        "empty_hypothesis_rate": empty_hypotheses / len(hypotheses),
        "empty_references": empty_references,
        "reference_length": {
            "mean": sum(len(reference) for reference in references) / len(references),
            "max": max(len(reference) for reference in references),
        },
        "hypothesis_length": {
            "mean": sum(len(hypothesis) for hypothesis in hypotheses) / len(hypotheses),
            "max": max(len(hypothesis) for hypothesis in hypotheses),
        },
        "distinct": distinct_ratio(hypotheses),
        "blank_ratio": (blank_steps / total_steps) if total_steps else 0.0,
        "top_predicted": used_tokens.most_common(15),
    }
    if vocabulary_size:
        metrics["vocabulary_size"] = vocabulary_size
        metrics["predicted_token_kinds"] = len(used_tokens)
        metrics["vocabulary_utilization"] = len(used_tokens) / vocabulary_size
    return metrics
