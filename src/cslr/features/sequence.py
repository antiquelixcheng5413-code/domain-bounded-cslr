from __future__ import annotations

from collections.abc import Sequence


def resample_indices(source_length: int, target_length: int) -> list[int]:
    if source_length <= 0:
        raise ValueError("source_length must be positive")
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    if target_length == 1:
        return [0]
    return [
        round(position * (source_length - 1) / (target_length - 1))
        for position in range(target_length)
    ]


def resample_sequence(sequence: Sequence[Sequence[float]], target_length: int) -> list[list[float]]:
    if not sequence:
        raise ValueError("sequence must not be empty")
    indices = resample_indices(len(sequence), target_length)
    return [list(sequence[index]) for index in indices]
