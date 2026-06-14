from __future__ import annotations

from collections.abc import Iterable


def classification_metrics(
    expected: Iterable[str], predicted: Iterable[str]
) -> dict[str, object]:
    y_true = list(expected)
    y_pred = list(predicted)
    if len(y_true) != len(y_pred):
        raise ValueError("expected and predicted lengths differ")
    if not y_true:
        raise ValueError("at least one prediction is required")

    labels = sorted(set(y_true) | set(y_pred))
    matrix = {actual: {guess: 0 for guess in labels} for actual in labels}
    for actual, guess in zip(y_true, y_pred, strict=False):
        matrix[actual][guess] += 1

    per_class: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    for label in labels:
        true_positive = matrix[label][label]
        false_positive = sum(matrix[actual][label] for actual in labels if actual != label)
        false_negative = sum(matrix[label][guess] for guess in labels if guess != label)
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
        f1_values.append(f1)

    correct = sum(1 for actual, guess in zip(y_true, y_pred, strict=False) if actual == guess)
    return {
        "accuracy": correct / len(y_true),
        "macro_f1": sum(f1_values) / len(f1_values),
        "labels": labels,
        "per_class": per_class,
        "confusion_matrix": [
            [matrix[actual][guess] for guess in labels] for actual in labels
        ],
    }
