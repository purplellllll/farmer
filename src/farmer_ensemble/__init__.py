"""Leakage-aware tabular ensemble training for Kaggriculture policy helpers.

The package intentionally models bounded decisions such as expert routing,
legal-candidate ranking, and auxiliary value targets.  It is not an API for
flattening the complete joint game action into a single unconstrained class.
"""

from .config import EnsembleConfig, ModelConfig
from .ensemble import EnsembleBundle, TrainingResult, load_bundle, train_ensemble

__all__ = [
    "EnsembleBundle",
    "EnsembleConfig",
    "ModelConfig",
    "TrainingResult",
    "load_bundle",
    "train_ensemble",
]
