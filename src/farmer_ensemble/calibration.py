from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


@dataclass
class TemperatureCalibrator:
    temperature_: float = 1.0

    def fit(
        self,
        probabilities: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "TemperatureCalibrator":
        p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0)
        labels = np.asarray(y, dtype=np.int64)
        weights = (
            np.ones(len(labels), dtype=np.float64)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float64)
        )
        weights = weights / weights.sum()
        logits = np.log(p)
        candidates = np.exp(np.linspace(np.log(0.2), np.log(5.0), 161))
        losses = []
        for temperature in candidates:
            calibrated = _softmax(logits / temperature)
            losses.append(-np.sum(weights * np.log(calibrated[np.arange(len(labels)), labels])))
        self.temperature_ = float(candidates[int(np.argmin(losses))])
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        logits = np.log(np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0))
        return _softmax(logits / self.temperature_)


@dataclass
class IdentityCalibrator:
    def fit(
        self,
        probabilities: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "IdentityCalibrator":
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, None)
        return p / p.sum(axis=1, keepdims=True)


def make_calibrator(method: str) -> TemperatureCalibrator | IdentityCalibrator:
    if method == "temperature":
        return TemperatureCalibrator()
    if method == "none":
        return IdentityCalibrator()
    raise ValueError(f"unknown calibration method: {method}")


def cross_fit_calibration(
    probabilities: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    method: str,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, TemperatureCalibrator | IdentityCalibrator]:
    calibrated = np.empty_like(probabilities, dtype=np.float64)
    for train_index, valid_index in folds:
        fold_calibrator = make_calibrator(method)
        fold_weights = None if sample_weight is None else sample_weight[train_index]
        fold_calibrator.fit(probabilities[train_index], y[train_index], fold_weights)
        calibrated[valid_index] = fold_calibrator.transform(probabilities[valid_index])
    final_calibrator = make_calibrator(method)
    final_calibrator.fit(probabilities, y, sample_weight)
    return calibrated, final_calibrator
