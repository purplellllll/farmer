from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from farmer_ensemble.config import EnsembleConfig, ModelConfig, SELECTED_MODELS
from farmer_ensemble.ensemble import load_bundle, train_ensemble
from farmer_ensemble.splitting import make_grouped_folds


def make_dataset() -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(123)
    n_groups, rows_per_group, n_features = 30, 4, 9
    groups = np.repeat(np.arange(n_groups), rows_per_group)
    seeds = groups + 10_000
    seats = np.tile(np.asarray([0, 1, 0, 1]), n_groups)
    X = rng.normal(size=(len(groups), n_features)).astype(np.float32)
    X[:, 0] += (groups % 3) * 0.5
    logits = np.column_stack(
        [X[:, 0] - 0.2 * X[:, 1], X[:, 1] + 0.3 * X[:, 2], -X[:, 0] - X[:, 1]]
    )
    y = logits.argmax(axis=1)
    # Guarantee each class occurs in many independent groups.
    for group in range(n_groups):
        y[group * rows_per_group] = group % 3
    feature_names = np.asarray([f"f_{index}" for index in range(n_features)])
    return X, y, groups, seeds, seats, feature_names


def smoke_config() -> EnsembleConfig:
    return EnsembleConfig(
        models=[
            ModelConfig("hgbc", params={"max_iter": 15, "min_samples_leaf": 4}),
            ModelConfig("extra_trees", params={"n_estimators": 18, "max_depth": 7}),
            ModelConfig("logistic_regression", params={"max_iter": 200}),
        ],
        n_splits=3,
        random_state=9,
        blend_iterations=40,
    )


class EnsembleSmokeTests(unittest.TestCase):
    def test_registry_has_exactly_selected_eight(self) -> None:
        self.assertEqual(
            SELECTED_MODELS,
            (
                "lightgbm",
                "xgboost",
                "catboost",
                "hgbc",
                "extra_trees",
                "logistic_regression",
                "ft_transformer",
                "realmlp",
            ),
        )

    def test_group_seed_seat_split_has_no_leakage(self) -> None:
        X, y, groups, seeds, seats, _ = make_dataset()
        result = make_grouped_folds(y, groups, seeds, seats, 3, 8, group_by_seed=True)
        self.assertEqual(len(result.fold_ids), len(X))
        self.assertTrue(all(item["group_overlap"] == 0 for item in result.audit["folds"]))
        self.assertTrue(all(item["seed_overlap"] == 0 for item in result.audit["folds"]))
        self.assertTrue(result.audit["seat_audit"]["mirrored_seats_kept_together"])

    def test_sklearn_only_train_predict_save_load(self) -> None:
        X, y, groups, seeds, seats, feature_names = make_dataset()
        result = train_ensemble(
            X,
            y,
            groups=groups,
            seeds=seeds,
            seats=seats,
            feature_names=feature_names.tolist(),
            config=smoke_config(),
        )
        self.assertEqual({report["status"] for report in result.model_reports}, {"trained"})
        probabilities = result.bundle.predict_proba(X[:7])
        self.assertEqual(probabilities.shape, (7, 3))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
        self.assertGreaterEqual(
            set(result.metrics), {"blend_oof", "stack_oof", "hybrid_oof", "blend_weights"}
        )

        with tempfile.TemporaryDirectory() as directory:
            manifest = result.save(directory)
            self.assertGreater(manifest["artifact"]["bytes"], 0)
            restored = load_bundle(directory)
            np.testing.assert_allclose(restored.predict_proba(X[:7]), probabilities)

    def test_missing_optional_dependency_is_reported(self) -> None:
        X, y, groups, seeds, seats, feature_names = make_dataset()
        import farmer_ensemble.adapters as adapters

        original = adapters._import_attribute

        def reject_lightgbm(module_name: str, attribute: str, install_hint: str):
            if module_name == "lightgbm":
                raise adapters.OptionalDependencyError("install with `pip install lightgbm`")
            return original(module_name, attribute, install_hint)

        config = smoke_config()
        config.models.insert(0, ModelConfig("lightgbm"))
        with patch.object(adapters, "_import_attribute", side_effect=reject_lightgbm):
            result = train_ensemble(
                X,
                y,
                groups=groups,
                seeds=seeds,
                seats=seats,
                feature_names=feature_names.tolist(),
                config=config,
            )
        report = next(item for item in result.model_reports if item["name"] == "lightgbm")
        self.assertEqual(report["status"], "skipped_missing_dependency")
        self.assertIn("pip install lightgbm", report["reason"])


if __name__ == "__main__":
    unittest.main()
