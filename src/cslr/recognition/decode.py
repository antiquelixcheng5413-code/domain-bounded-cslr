"""CTC decoding: greedy collapse and prefix beam search (no external language model)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cslr.recognition.model import BLANK_INDEX


@dataclass(frozen=True)
class DecodedSequence:
    classes: tuple[int, ...]
    score: float


def greedy_decode(log_probs: np.ndarray) -> DecodedSequence:
    """Collapse repeats and drop blanks from the argmax path."""

    if log_probs.ndim != 2:
        raise ValueError(f"expected [T, C] log probabilities, got {log_probs.shape}")
    best = log_probs.argmax(axis=1)
    collapsed: list[int] = []
    score = 0.0
    previous = None
    for step, index in enumerate(best):
        score += float(log_probs[step, index])
        current = int(index)
        if current != previous and current != BLANK_INDEX:
            collapsed.append(current)
        previous = current
    return DecodedSequence(classes=tuple(collapsed), score=score)


def prefix_beam_search(
    log_probs: np.ndarray,
    beam_width: int = 10,
    length_penalty: float = 0.0,
) -> DecodedSequence:
    """Standard CTC prefix beam search.

    ``log_probs`` is ``[T, C]`` with index ``BLANK_INDEX`` reserved for blank.
    ``length_penalty`` (alpha) favours longer hypotheses when > 0; the default
    keeps the plain acoustic score so results stay comparable with greedy.
    """

    if log_probs.ndim != 2:
        raise ValueError(f"expected [T, C] log probabilities, got {log_probs.shape}")
    if beam_width < 1:
        raise ValueError("beam_width must be at least 1")

    # prefix -> (log p of paths ending in blank, log p of paths ending in a label)
    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, -np.inf)}
    for step in range(log_probs.shape[0]):
        step_log_probs = log_probs[step]
        candidates: dict[tuple[int, ...], list[float]] = {}

        def add(prefix: tuple[int, ...], blank: float, non_blank: float) -> None:
            slot = candidates.setdefault(prefix, [-np.inf, -np.inf])
            slot[0] = float(np.logaddexp(slot[0], blank))
            slot[1] = float(np.logaddexp(slot[1], non_blank))

        for prefix, (log_p_blank, log_p_label) in beams.items():
            total = np.logaddexp(log_p_blank, log_p_label)
            add(prefix, total + step_log_probs[BLANK_INDEX], -np.inf)
            for index in range(log_probs.shape[1]):
                if index == BLANK_INDEX:
                    continue
                probability = step_log_probs[index]
                if prefix and prefix[-1] == index:
                    # repeat: stays on the same prefix (from blank) or extends it (from label)
                    add(prefix, -np.inf, log_p_blank + probability)
                    add(prefix + (index,), -np.inf, log_p_label + probability)
                else:
                    add(prefix + (index,), -np.inf, total + probability)

        ranked = sorted(
            candidates.items(),
            key=lambda item: float(np.logaddexp(item[1][0], item[1][1])),
            reverse=True,
        )[:beam_width]
        beams = {prefix: (values[0], values[1]) for prefix, values in ranked}

    if not beams:
        return DecodedSequence(classes=(), score=float("-inf"))

    def final_score(item: tuple[tuple[int, ...], tuple[float, float]]) -> float:
        prefix, (log_p_blank, log_p_label) = item
        score = float(np.logaddexp(log_p_blank, log_p_label))
        if length_penalty and prefix:
            score += length_penalty * len(prefix)
        return score

    best_prefix, best_values = max(beams.items(), key=lambda item: final_score(item))
    return DecodedSequence(
        classes=best_prefix,
        score=float(np.logaddexp(best_values[0], best_values[1])),
    )


def classes_to_token_ids(classes: tuple[int, ...]) -> list[int]:
    """Map CTC class indices back to vocabulary indices."""

    return [index - 1 for index in classes if index != BLANK_INDEX]


def log_prob_stats(log_probs: np.ndarray) -> dict[str, float]:
    """Compact diagnostics for a ``[T, C]`` log-probability block."""

    return {
        "steps": float(log_probs.shape[0]),
        "classes": float(log_probs.shape[1]),
        "min": float(log_probs.min()),
        "max": float(log_probs.max()),
        "sum_exp": float(np.exp(log_probs).sum()),
    }
