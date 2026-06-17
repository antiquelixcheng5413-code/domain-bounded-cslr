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


def multilabel_metrics(
    expected: Iterable[Iterable[str]],
    predicted: Iterable[Iterable[str]],
    labels: list[str],
) -> dict[str, object]:
    y_true = [set(tokens) for tokens in expected]
    y_pred = [set(tokens) for tokens in predicted]
    if len(y_true) != len(y_pred):
        raise ValueError("expected and predicted lengths differ")
    if not y_true:
        raise ValueError("at least one prediction is required")
    if not labels:
        raise ValueError("at least one label is required")

    per_label: dict[str, dict[str, float]] = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0
    f1_values: list[float] = []
    pairs = list(zip(y_true, y_pred, strict=False))
    for label in labels:
        tp = sum(1 for actual, guess in pairs if label in actual & guess)
        fp = sum(1 for actual, guess in pairs if label not in actual and label in guess)
        fn = sum(1 for actual, guess in pairs if label in actual and label not in guess)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(tp + fn),
        }
        f1_values.append(f1)

    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    subset_correct = sum(
        1 for actual, guess in zip(y_true, y_pred, strict=False) if actual == guess
    )
    return {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_f1": sum(f1_values) / len(f1_values),
        "subset_accuracy": subset_correct / len(y_true),
        "labels": labels,
        "per_label": per_label,
    }
