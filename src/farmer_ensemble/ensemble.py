from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import platform
import time
from typing import Any

import numpy as np

from .adapters import EstimatorAdapter, OptionalDependencyError, create_adapter
from .calibration import IdentityCalibrator, TemperatureCalibrator, cross_fit_calibration
from .config import EnsembleConfig, ModelConfig
from .metrics import classification_metrics
from .splitting import SplitResult, make_grouped_folds


Calibrator = TemperatureCalibrator | IdentityCalibrator


@dataclass
class FittedBaseModel:
    name: str
    adapter: EstimatorAdapter
    calibrator: Calibrator
    metrics: dict[str, Any]


@dataclass
class EnsembleBundle:
    classes_: np.ndarray
    feature_names: list[str]
    base_models: list[FittedBaseModel]
    blend_weights: np.ndarray
    stacker: Any
    config: EnsembleConfig

    def _base_probabilities(self, X: np.ndarray) -> list[np.ndarray]:
        return [
            item.calibrator.transform(item.adapter.predict_proba(X))
            for item in self.base_models
        ]

    def predict_proba(self, X: np.ndarray, mode: str | None = None) -> np.ndarray:
        mode = self.config.prediction if mode is None else mode
        base = self._base_probabilities(X)
        if not base:
            raise RuntimeError("bundle has no fitted base models")
        blend = np.tensordot(self.blend_weights, np.stack(base), axes=(0, 0))
        stack = _aligned_stacker_predict(self.stacker, np.concatenate(base, axis=1), len(self.classes_))
        if mode == "blend":
            result = blend
        elif mode == "stack":
            result = stack
        elif mode == "hybrid":
            alpha = self.config.hybrid_blend_weight
            result = alpha * blend + (1.0 - alpha) * stack
        else:
            raise ValueError(f"unknown prediction mode: {mode}")
        result = np.clip(result, 1e-12, None)
        return result / result.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray, mode: str | None = None) -> np.ndarray:
        return self.classes_[self.predict_proba(X, mode).argmax(axis=1)]


@dataclass
class TrainingResult:
    bundle: EnsembleBundle
    oof_probabilities: dict[str, np.ndarray]
    metrics: dict[str, Any]
    model_reports: list[dict[str, Any]]
    split_audit: dict[str, Any]
    feature_manifest: dict[str, Any]

    def save(self, output_dir: str | Path) -> dict[str, Any]:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        model_path = directory / "ensemble_bundle.pkl"
        with model_path.open("wb") as handle:
            pickle.dump(self.bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
        artifact_bytes = model_path.stat().st_size
        artifact_hash = _sha256(model_path)
        latency = benchmark_inference(self.bundle, self._benchmark_matrix())
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifact": {
                "file": model_path.name,
                "sha256": artifact_hash,
                "bytes": artifact_bytes,
                "mib": artifact_bytes / (1024**2),
                "budget_mib": self.bundle.config.model_size_budget_mib,
                "within_model_budget": artifact_bytes
                <= self.bundle.config.model_size_budget_mib * 1024**2,
            },
            "runtime": _runtime_manifest(),
            "models": self.model_reports,
            "metrics": self.metrics,
            "split_audit": self.split_audit,
            "inference_benchmark": latency,
            "config": self.bundle.config.to_dict(),
        }
        _write_json(directory / "manifest.json", manifest)
        _write_json(directory / "feature_manifest.json", self.feature_manifest)
        return manifest

    def _benchmark_matrix(self) -> np.ndarray:
        # The saved feature schema is sufficient for a structural latency smoke test.
        return np.zeros((1, len(self.bundle.feature_names)), dtype=np.float32)


def compute_sample_weight(y: np.ndarray, mode: str) -> np.ndarray | None:
    if mode == "none":
        return None
    labels = np.asarray(y, dtype=np.int64)
    counts = np.bincount(labels, minlength=int(labels.max()) + 1).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("encoded training labels must be contiguous")
    weights = len(labels) / (len(counts) * counts)
    if mode == "sqrt_balanced":
        weights = np.sqrt(weights)
    sample_weight = weights[labels]
    return sample_weight / sample_weight.mean()


def _project_simplex(values: np.ndarray) -> np.ndarray:
    sorted_values = np.sort(values)[::-1]
    cumulative = np.cumsum(sorted_values) - 1.0
    indices = np.arange(1, len(values) + 1)
    valid = sorted_values - cumulative / indices > 0
    rho = int(indices[valid][-1])
    theta = cumulative[rho - 1] / rho
    return np.maximum(values - theta, 0.0)


def fit_blend_weights(
    probabilities: list[np.ndarray],
    y: np.ndarray,
    iterations: int,
    learning_rate: float,
) -> np.ndarray:
    stacked = np.stack(probabilities, axis=0)
    labels = np.asarray(y, dtype=np.int64)
    weights = np.full(len(probabilities), 1.0 / len(probabilities), dtype=np.float64)
    true_probabilities = stacked[:, np.arange(len(labels)), labels]
    for step in range(iterations):
        blended_true = np.clip(weights @ true_probabilities, 1e-12, None)
        gradient = -np.mean(true_probabilities / blended_true[None, :], axis=1)
        # Decay makes the routine stable without requiring scipy.
        rate = learning_rate / np.sqrt(1.0 + step / 25.0)
        weights = _project_simplex(weights - rate * gradient)
    return weights / weights.sum()


def _make_stacker(config: EnsembleConfig, random_state: int) -> Any:
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(
        C=config.stacker_c,
        max_iter=600,
        solver="lbfgs",
        random_state=random_state,
    )


def _aligned_stacker_predict(stacker: Any, X: np.ndarray, n_classes: int) -> np.ndarray:
    raw = np.asarray(stacker.predict_proba(X), dtype=np.float64)
    result = np.zeros((len(X), n_classes), dtype=np.float64)
    for index, label in enumerate(np.asarray(stacker.classes_)):
        result[:, int(label)] = raw[:, index]
    result = np.clip(result, 1e-12, None)
    return result / result.sum(axis=1, keepdims=True)


def _cross_fit_meta_models(
    base_oof: list[np.ndarray],
    y: np.ndarray,
    split: SplitResult,
    config: EnsembleConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    stack_features = np.concatenate(base_oof, axis=1)
    blend_oof = np.empty_like(base_oof[0])
    stack_oof = np.empty_like(base_oof[0])
    for fold_index, (train_index, valid_index) in enumerate(split.folds):
        fold_probabilities = [p[train_index] for p in base_oof]
        weights = fit_blend_weights(
            fold_probabilities,
            y[train_index],
            config.blend_iterations,
            config.blend_learning_rate,
        )
        blend_oof[valid_index] = np.tensordot(
            weights,
            np.stack([p[valid_index] for p in base_oof]),
            axes=(0, 0),
        )
        stacker = _make_stacker(config, config.random_state + 7000 + fold_index)
        stacker.fit(stack_features[train_index], y[train_index])
        stack_oof[valid_index] = _aligned_stacker_predict(
            stacker, stack_features[valid_index], base_oof[0].shape[1]
        )
    final_weights = fit_blend_weights(
        base_oof, y, config.blend_iterations, config.blend_learning_rate
    )
    final_stacker = _make_stacker(config, config.random_state + 8000)
    final_stacker.fit(stack_features, y)
    return blend_oof, stack_oof, final_weights, final_stacker


def train_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    *,
    groups: np.ndarray | None,
    seeds: np.ndarray | None = None,
    seats: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    config: EnsembleConfig | None = None,
) -> TrainingResult:
    config = EnsembleConfig() if config is None else config
    config.validate()
    values = np.asarray(X, dtype=np.float32)
    original_labels = np.asarray(y)
    if values.ndim != 2 or len(values) != len(original_labels):
        raise ValueError("X must be 2D and have the same number of rows as y")
    classes, encoded = np.unique(original_labels, return_inverse=True)
    if len(classes) < 2:
        raise ValueError("at least two target classes are required")
    names = feature_names or [f"feature_{index}" for index in range(values.shape[1])]
    if len(names) != values.shape[1] or len(set(names)) != len(names):
        raise ValueError("feature_names must be unique and match X columns")

    split = make_grouped_folds(
        encoded,
        groups,
        seeds,
        seats,
        config.n_splits,
        config.random_state,
        config.group_by_seed,
    )
    global_weight = compute_sample_weight(encoded, config.class_balance)
    fitted: list[FittedBaseModel] = []
    base_oof: list[np.ndarray] = []
    reports: list[dict[str, Any]] = []
    oof_by_name: dict[str, np.ndarray] = {}

    for model_index, model_config in enumerate(config.models):
        if not model_config.enabled:
            reports.append({"name": model_config.name, "status": "disabled"})
            continue
        started = time.perf_counter()
        try:
            raw_oof = np.empty((len(values), len(classes)), dtype=np.float64)
            fold_weight_modes: list[str] = []
            for fold_index, (train_index, valid_index) in enumerate(split.folds):
                adapter = create_adapter(
                    model_config,
                    len(classes),
                    config.random_state + model_index * 100 + fold_index,
                    config.unsupported_weight_strategy,
                )
                fold_weight = compute_sample_weight(encoded[train_index], config.class_balance)
                adapter.fit(values[train_index], encoded[train_index], fold_weight)
                raw_oof[valid_index] = adapter.predict_proba(values[valid_index])
                fold_weight_modes.append(adapter.fit_metadata["weight_handling"])
            calibrated_oof, calibrator = cross_fit_calibration(
                raw_oof,
                encoded,
                split.folds,
                config.calibration,
                # Calibrate back to the empirical deployment prior. Class weights
                # belong to estimator training, not probability calibration.
                None,
            )
            final_adapter = create_adapter(
                model_config,
                len(classes),
                config.random_state + model_index * 100 + 99,
                config.unsupported_weight_strategy,
            )
            final_adapter.fit(values, encoded, global_weight)
            model_metrics = {
                "raw_oof": classification_metrics(encoded, raw_oof),
                "calibrated_oof": classification_metrics(encoded, calibrated_oof),
            }
            fitted.append(
                FittedBaseModel(model_config.name, final_adapter, calibrator, model_metrics)
            )
            base_oof.append(calibrated_oof)
            oof_by_name[model_config.name] = calibrated_oof
            reports.append(
                {
                    "name": model_config.name,
                    "status": "trained",
                    "seconds": time.perf_counter() - started,
                    "fold_weight_handling": sorted(set(fold_weight_modes)),
                    "final_weight_handling": final_adapter.fit_metadata["weight_handling"],
                    "metrics": model_metrics,
                }
            )
        except OptionalDependencyError as exc:
            reports.append(
                {
                    "name": model_config.name,
                    "status": "skipped_missing_dependency",
                    "reason": str(exc),
                }
            )
        except Exception as exc:
            if not config.skip_failed_models:
                raise
            reports.append(
                {
                    "name": model_config.name,
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

    if not fitted:
        reasons = "; ".join(f"{r['name']}: {r.get('reason', r['status'])}" for r in reports)
        raise RuntimeError(f"no estimator could be trained ({reasons})")

    blend_oof, stack_oof, blend_weights, stacker = _cross_fit_meta_models(
        base_oof, encoded, split, config
    )
    alpha = config.hybrid_blend_weight
    hybrid_oof = alpha * blend_oof + (1.0 - alpha) * stack_oof
    oof_by_name.update(blend=blend_oof, stack=stack_oof, hybrid=hybrid_oof)
    metrics = {
        "blend_oof": classification_metrics(encoded, blend_oof),
        "stack_oof": classification_metrics(encoded, stack_oof),
        "hybrid_oof": classification_metrics(encoded, hybrid_oof),
        "blend_weights": {
            model.name: float(weight) for model, weight in zip(fitted, blend_weights)
        },
    }
    bundle = EnsembleBundle(
        classes_=classes,
        feature_names=list(names),
        base_models=fitted,
        blend_weights=blend_weights,
        stacker=stacker,
        config=config,
    )
    feature_manifest = {
        "schema_version": 1,
        "n_features": values.shape[1],
        "feature_names_in_order": list(names),
        "input_dtype": "float32-compatible numeric",
        "non_finite_policy": "replace inf with NaN, then training-fold median imputation",
        "target_classes": [_json_scalar(value) for value in classes],
        "target_semantics": (
            "bounded expert route, legal candidate, or auxiliary value class; "
            "never an unconstrained flattened joint action"
        ),
    }
    return TrainingResult(
        bundle=bundle,
        oof_probabilities=oof_by_name,
        metrics=metrics,
        model_reports=reports,
        split_audit=split.audit,
        feature_manifest=feature_manifest,
    )


def benchmark_inference(
    bundle: EnsembleBundle, X: np.ndarray, repetitions: int = 10
) -> dict[str, Any]:
    values = np.asarray(X, dtype=np.float32)
    bundle.predict_proba(values)
    durations = []
    for _ in range(repetitions):
        started = time.perf_counter()
        bundle.predict_proba(values)
        durations.append((time.perf_counter() - started) * 1000.0)
    return {
        "rows_per_call": len(values),
        "repetitions": repetitions,
        "mean_ms": float(np.mean(durations)),
        "p95_ms": float(np.percentile(durations, 95)),
        "one_second_target_met": bool(np.percentile(durations, 95) < 1000.0),
        "note": "local structural benchmark; repeat inside the Kaggle image on real states",
    }


def load_bundle(path: str | Path, verify_manifest: bool = True) -> EnsembleBundle:
    model_path = Path(path)
    if model_path.is_dir():
        model_path = model_path / "ensemble_bundle.pkl"
    if verify_manifest:
        manifest_path = model_path.parent / "manifest.json"
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            expected = manifest.get("artifact", {}).get("sha256")
            if expected and _sha256(model_path) != expected:
                raise ValueError("model artifact hash does not match manifest")
    with model_path.open("rb") as handle:
        bundle = pickle.load(handle)
    if not isinstance(bundle, EnsembleBundle):
        raise TypeError("artifact does not contain an EnsembleBundle")
    return bundle


def _runtime_manifest() -> dict[str, Any]:
    packages = {}
    for package in ("numpy", "scikit-learn", "lightgbm", "xgboost", "catboost", "pytabkit"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _json_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value
