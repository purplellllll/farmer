from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


SELECTED_MODELS = (
    "lightgbm",
    "xgboost",
    "catboost",
    "hgbc",
    "extra_trees",
    "logistic_regression",
    "ft_transformer",
    "realmlp",
)


@dataclass(slots=True)
class ModelConfig:
    name: str
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModelConfig":
        return cls(
            name=str(value["name"]),
            enabled=bool(value.get("enabled", True)),
            params=dict(value.get("params", {})),
        )


@dataclass(slots=True)
class EnsembleConfig:
    models: list[ModelConfig] = field(
        default_factory=lambda: [ModelConfig(name=name) for name in SELECTED_MODELS]
    )
    n_splits: int = 5
    random_state: int = 20260825
    class_balance: str = "sqrt_balanced"
    unsupported_weight_strategy: str = "weighted_resample"
    calibration: str = "temperature"
    prediction: str = "hybrid"
    hybrid_blend_weight: float = 0.5
    stacker_c: float = 1.0
    blend_iterations: int = 400
    blend_learning_rate: float = 0.05
    group_by_seed: bool = True
    skip_failed_models: bool = True
    model_size_budget_mib: float = 95.0

    def validate(self) -> None:
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if self.class_balance not in {"none", "balanced", "sqrt_balanced"}:
            raise ValueError("unsupported class_balance")
        if self.unsupported_weight_strategy not in {"error", "weighted_resample"}:
            raise ValueError("unsupported unsupported_weight_strategy")
        if self.calibration not in {"none", "temperature"}:
            raise ValueError("unsupported calibration")
        if self.prediction not in {"blend", "stack", "hybrid"}:
            raise ValueError("prediction must be blend, stack, or hybrid")
        if not 0.0 <= self.hybrid_blend_weight <= 1.0:
            raise ValueError("hybrid_blend_weight must be in [0, 1]")
        names = [m.name for m in self.models if m.enabled]
        unknown = sorted(set(names) - set(SELECTED_MODELS))
        if unknown:
            raise ValueError(f"unknown models: {unknown}")
        if len(names) != len(set(names)):
            raise ValueError("model names must be unique")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EnsembleConfig":
        kwargs = dict(value)
        if "models" in value:
            kwargs["models"] = [ModelConfig.from_dict(v) for v in value["models"]]
        result = cls(**kwargs)
        result.validate()
        return result

    @classmethod
    def load(cls, path: str | Path) -> "EnsembleConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
