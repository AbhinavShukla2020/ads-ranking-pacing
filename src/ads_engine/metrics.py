from __future__ import annotations

import math
from collections.abc import Sequence


def binary_log_loss(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    """Return mean binary cross-entropy with numerically safe clipping."""
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("labels and probabilities must have the same non-zero length")

    total = 0.0
    for label, probability in zip(labels, probabilities, strict=True):
        if label not in (0, 1):
            raise ValueError("labels must contain only 0 or 1")
        p = min(max(float(probability), 1e-12), 1.0 - 1e-12)
        total -= label * math.log(p) + (1 - label) * math.log(1.0 - p)
    return total / len(labels)


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute ROC AUC from average ranks, including tied-score handling."""
    if len(labels) != len(scores) or not labels:
        raise ValueError("labels and scores must have the same non-zero length")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("labels must contain only 0 or 1")

    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("ROC AUC requires both positive and negative examples")

    ordered = sorted(zip(scores, labels, strict=True), key=lambda pair: pair[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        index = end

    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )
