from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EditCounts:
    substitutions: int
    deletions: int
    insertions: int
    reference_tokens: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions


@dataclass(frozen=True)
class CTCDecodedSequence:
    tokens: list[str]
    path_score: float


def ctc_greedy_decode(
    logits: np.ndarray,
    index_to_token: dict[int, str],
    blank_index: int,
) -> CTCDecodedSequence:
    if logits.ndim != 2:
        raise ValueError(f"expected [frames, classes] logits, got {logits.shape}")
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    indices = probabilities.argmax(axis=1)
    selected = probabilities[np.arange(len(indices)), indices]
    tokens: list[str] = []
    previous: int | None = None
    selected_log_probs: list[float] = []
    for index, probability in zip(indices.tolist(), selected.tolist(), strict=False):
        if index == blank_index:
            previous = None
            continue
        if index != previous:
            token = index_to_token.get(index)
            if token is not None:
                tokens.append(token)
                selected_log_probs.append(math.log(max(probability, 1e-12)))
        previous = index
    path_score = (
        math.exp(sum(selected_log_probs) / len(selected_log_probs))
        if selected_log_probs
        else 0.0
    )
    return CTCDecodedSequence(tokens=tokens, path_score=path_score)


def token_edit_counts(reference: list[str], hypothesis: list[str]) -> EditCounts:
    rows = len(reference) + 1
    columns = len(hypothesis) + 1
    table: list[list[tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0) for _ in range(columns)] for _ in range(rows)
    ]
    for row in range(1, rows):
        table[row][0] = (row, 0, row, 0)
    for column in range(1, columns):
        table[0][column] = (column, 0, 0, column)
    for row in range(1, rows):
        for column in range(1, columns):
            if reference[row - 1] == hypothesis[column - 1]:
                table[row][column] = table[row - 1][column - 1]
                continue
            substitution = table[row - 1][column - 1]
            deletion = table[row - 1][column]
            insertion = table[row][column - 1]
            choices = [
                (substitution[0] + 1, substitution[1] + 1, substitution[2], substitution[3]),
                (deletion[0] + 1, deletion[1], deletion[2] + 1, deletion[3]),
                (insertion[0] + 1, insertion[1], insertion[2], insertion[3] + 1),
            ]
            table[row][column] = min(choices, key=lambda value: value[0])
    _, substitutions, deletions, insertions = table[-1][-1]
    return EditCounts(
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_tokens=len(reference),
    )


def corpus_wer(counts: Iterable[EditCounts]) -> dict[str, float | int]:
    values = list(counts)
    substitutions = sum(value.substitutions for value in values)
    deletions = sum(value.deletions for value in values)
    insertions = sum(value.insertions for value in values)
    reference_tokens = sum(value.reference_tokens for value in values)
    errors = substitutions + deletions + insertions
    return {
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "reference_tokens": reference_tokens,
        "errors": errors,
        "wer": errors / reference_tokens if reference_tokens else 0.0,
    }
