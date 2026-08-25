from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import inspect
from typing import Any

import numpy as np

from .config import ModelConfig


class OptionalDependencyError(RuntimeError):
    """Raised when an explicitly selected optional estimator is unavailable."""


@dataclass
class NumericPreprocessor:
    scale: bool = False
    median_: np.ndarray | None = None
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "NumericPreprocessor":
        values = self._as_float(X)
        with np.errstate(all="ignore"):
            medians = np.nanmedian(values, axis=0)
        self.median_ = np.where(np.isfinite(medians), medians, 0.0).astype(np.float32)
        clean = self._impute(values)
        if self.scale:
            self.mean_ = clean.mean(axis=0, dtype=np.float64).astype(np.float32)
            std = clean.std(axis=0, dtype=np.float64)
            self.std_ = np.where(std > 1e-8, std, 1.0).astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.median_ is None:
            raise RuntimeError("preprocessor is not fitted")
        clean = self._impute(self._as_float(X))
        if self.scale:
            assert self.mean_ is not None and self.std_ is not None
            clean = (clean - self.mean_) / self.std_
        return np.asarray(clean, dtype=np.float32, order="C")

    @staticmethod
    def _as_float(X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"X must be two-dimensional, got {values.shape}")
        return np.where(np.isfinite(values), values, np.nan)

    def _impute(self, values: np.ndarray) -> np.ndarray:
        assert self.median_ is not None
        return np.where(np.isnan(values), self.median_[None, :], values)


@dataclass
class EstimatorAdapter:
    name: str
    estimator: Any
    n_classes: int
    random_state: int
    scale_features: bool = False
    unsupported_weight_strategy: str = "weighted_resample"
    preprocessor: NumericPreprocessor = field(init=False)
    fit_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.preprocessor = NumericPreprocessor(scale=self.scale_features)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "EstimatorAdapter":
        values = self.preprocessor.fit(X).transform(X)
        labels = np.asarray(y, dtype=np.int64)
        supports_weight = _fit_accepts(self.estimator, "sample_weight")
        if sample_weight is None or supports_weight:
            kwargs = {} if sample_weight is None else {"sample_weight": sample_weight}
            self.estimator.fit(values, labels, **kwargs)
            self.fit_metadata["weight_handling"] = (
                "none" if sample_weight is None else "sample_weight"
            )
            return self

        if self.unsupported_weight_strategy == "error":
            raise TypeError(f"{self.name} does not accept sample_weight")
        values, labels = _weighted_resample(
            values, labels, np.asarray(sample_weight), self.random_state
        )
        self.estimator.fit(values, labels)
        self.fit_metadata["weight_handling"] = "weighted_resample"
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        values = self.preprocessor.transform(X)
        probabilities = np.asarray(self.estimator.predict_proba(values), dtype=np.float64)
        if probabilities.ndim == 1:
            probabilities = np.column_stack([1.0 - probabilities, probabilities])
        classes = np.asarray(getattr(self.estimator, "classes_", np.arange(probabilities.shape[1])))
        aligned = np.zeros((len(values), self.n_classes), dtype=np.float64)
        for source_index, label in enumerate(classes):
            class_index = int(label)
            if 0 <= class_index < self.n_classes:
                aligned[:, class_index] = probabilities[:, source_index]
        aligned = np.clip(aligned, 1e-12, None)
        return aligned / aligned.sum(axis=1, keepdims=True)


def _fit_accepts(estimator: Any, parameter: str) -> bool:
    try:
        signature = inspect.signature(estimator.fit)
    except (TypeError, ValueError):
        return False
    return parameter in signature.parameters


def _weighted_resample(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(sample_weight, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(y), size=len(y), replace=True, p=probabilities)
    # Ensure every original class remains trainable even for a tiny fold.
    present = set(y[indices].tolist())
    missing = [label for label in np.unique(y) if label not in present]
    if missing:
        indices[: len(missing)] = [int(np.flatnonzero(y == label)[0]) for label in missing]
    return X[indices], y[indices]


def _import_attribute(module_name: str, attribute: str, install_hint: str) -> Any:
    try:
        module = importlib.import_module(module_name)
        return getattr(module, attribute)
    except (ImportError, AttributeError) as exc:
        raise OptionalDependencyError(
            f"missing {attribute}; install with `{install_hint}`"
        ) from exc


def create_adapter(
    config: ModelConfig,
    n_classes: int,
    random_state: int,
    unsupported_weight_strategy: str = "weighted_resample",
) -> EstimatorAdapter:
    name = config.name
    params = dict(config.params)
    scale = False

    if name == "hgbc":
        from sklearn.ensemble import HistGradientBoostingClassifier

        defaults = dict(
            learning_rate=0.06,
            max_iter=180,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=random_state,
        )
        defaults.update(params)
        estimator = HistGradientBoostingClassifier(**defaults)
    elif name == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier

        defaults = dict(
            n_estimators=160,
            max_depth=16,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=1,
            random_state=random_state,
        )
        defaults.update(params)
        estimator = ExtraTreesClassifier(**defaults)
    elif name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        defaults = dict(C=1.0, max_iter=500, solver="lbfgs", random_state=random_state)
        defaults.update(params)
        estimator = LogisticRegression(**defaults)
        scale = True
    elif name == "lightgbm":
        LGBMClassifier = _import_attribute(
            "lightgbm", "LGBMClassifier", "pip install lightgbm"
        )
        defaults = dict(
            n_estimators=180,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=10,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            n_jobs=1,
            verbosity=-1,
            random_state=random_state,
        )
        defaults.update(params)
        estimator = LGBMClassifier(**defaults)
    elif name == "xgboost":
        XGBClassifier = _import_attribute(
            "xgboost", "XGBClassifier", "pip install xgboost"
        )
        defaults = dict(
            n_estimators=180,
            learning_rate=0.05,
            max_depth=7,
            min_child_weight=2.0,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            tree_method="hist",
            objective="multi:softprob" if n_classes > 2 else "binary:logistic",
            eval_metric="mlogloss" if n_classes > 2 else "logloss",
            n_jobs=1,
            random_state=random_state,
        )
        defaults.update(params)
        estimator = XGBClassifier(**defaults)
    elif name == "catboost":
        CatBoostClassifier = _import_attribute(
            "catboost", "CatBoostClassifier", "pip install catboost"
        )
        defaults = dict(
            iterations=180,
            learning_rate=0.05,
            depth=7,
            l2_leaf_reg=3.0,
            loss_function="MultiClass" if n_classes > 2 else "Logloss",
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
            random_seed=random_state,
        )
        defaults.update(params)
        estimator = CatBoostClassifier(**defaults)
    elif name == "ft_transformer":
        FTTClassifier = _import_attribute(
            "pytabkit", "FTT_D_Classifier", "pip install pytabkit torch"
        )
        defaults = dict(
            device="cpu",
            random_state=random_state,
            n_cv=1,
            n_refit=0,
            n_threads=1,
            max_epochs=64,
            batch_size=256,
            verbosity=0,
        )
        defaults.update(params)
        estimator = FTTClassifier(**defaults)
        scale = False  # PyTabKit owns the feature preprocessing.
    elif name == "realmlp":
        RealMLPClassifier = _import_attribute(
            "pytabkit", "RealMLP_TD_Classifier", "pip install pytabkit torch"
        )
        defaults = dict(
            device="cpu",
            random_state=random_state,
            n_cv=1,
            n_refit=0,
            n_threads=1,
            n_epochs=64,
            batch_size=256,
            val_metric_name="cross_entropy",
            use_ls=False,
            verbosity=0,
        )
        defaults.update(params)
        estimator = RealMLPClassifier(**defaults)
        scale = False
    else:
        raise ValueError(f"unknown estimator: {name}")

    return EstimatorAdapter(
        name=name,
        estimator=estimator,
        n_classes=n_classes,
        random_state=random_state,
        scale_features=scale,
        unsupported_weight_strategy=unsupported_weight_strategy,
    )
