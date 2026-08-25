from __future__ import annotations

import numpy as np


def expected_calibration_error(
    y: np.ndarray, probabilities: np.ndarray, n_bins: int = 15
) -> float:
    labels = np.asarray(y, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    confidence = p.max(axis=1)
    correct = (p.argmax(axis=1) == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if np.any(mask):
            error += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(error)


def classification_metrics(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss

    labels = np.asarray(y, dtype=np.int64)
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0)
    predictions = p.argmax(axis=1)
    one_hot = np.eye(p.shape[1], dtype=np.float64)[labels]
    return {
        "log_loss": float(log_loss(labels, p, labels=np.arange(p.shape[1]))),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "multiclass_brier": float(np.mean(np.sum((p - one_hot) ** 2, axis=1))),
        "ece_15": expected_calibration_error(labels, p, 15),
    }
