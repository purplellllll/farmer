"""Kaggle GPU v4: leakage-safe complete-action eight-model ensemble."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


os.environ.setdefault("PYTHONUNBUFFERED", "1")
started = time.perf_counter()

# The P100 image needs a wheel that still ships sm_60 kernels.
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "torch==2.7.1",
        "torchvision==0.22.1",
        "torchaudio==2.7.1",
        "--index-url",
        "https://download.pytorch.org/whl/cu126",
    ]
)
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "git+https://github.com/purplellllll/farmer.git@codex/rl-ensemble-foundation",
        "pytabkit>=1.3",
        "skorch>=1.0",
    ]
)

import numpy as np
import torch
from sklearn.metrics import recall_score

from farmer_ensemble.config import EnsembleConfig
from farmer_ensemble.ensemble import train_ensemble


data_paths = list(Path("/kaggle/input").rglob("joint_router_train.npz"))
if len(data_paths) != 1:
    raise RuntimeError(f"expected one joint_router_train.npz, found {data_paths}")

config = EnsembleConfig.from_dict(
    {
        "n_splits": 4,
        "random_state": 20260826,
        "class_balance": "sqrt_balanced",
        "unsupported_weight_strategy": "weighted_resample",
        "calibration": "temperature",
        "prediction": "hybrid",
        "hybrid_blend_weight": 0.55,
        "stacker_c": 0.7,
        "blend_iterations": 500,
        "blend_learning_rate": 0.035,
        "group_by_seed": True,
        "skip_failed_models": False,
        "model_size_budget_mib": 95.0,
        "models": [
            {
                "name": "lightgbm",
                "params": {
                    "n_estimators": 280,
                    "learning_rate": 0.035,
                    "num_leaves": 63,
                    "max_depth": 12,
                    "min_child_samples": 30,
                    "n_jobs": 4,
                },
            },
            {
                "name": "xgboost",
                "params": {
                    "n_estimators": 280,
                    "learning_rate": 0.035,
                    "max_depth": 9,
                    "min_child_weight": 3.0,
                    "tree_method": "hist",
                    "device": "cuda",
                    "n_jobs": 4,
                },
            },
            {
                "name": "catboost",
                "params": {
                    "iterations": 280,
                    "learning_rate": 0.04,
                    "depth": 9,
                    "task_type": "GPU",
                    "devices": "0",
                    "thread_count": 4,
                },
            },
            {
                "name": "hgbc",
                "params": {
                    "max_iter": 240,
                    "max_leaf_nodes": 63,
                    "min_samples_leaf": 30,
                    "l2_regularization": 1.5,
                },
            },
            {
                "name": "extra_trees",
                "params": {
                    "n_estimators": 320,
                    "max_depth": 22,
                    "min_samples_leaf": 2,
                    "max_features": 0.8,
                    "n_jobs": 4,
                },
            },
            {
                "name": "logistic_regression",
                "params": {"C": 0.7, "max_iter": 1000},
            },
            {
                "name": "ft_transformer",
                "params": {
                    "device": "cuda",
                    "max_epochs": 36,
                    "batch_size": 512,
                    "n_threads": 4,
                },
            },
            {
                "name": "realmlp",
                "params": {
                    "device": "cuda",
                    "n_epochs": 36,
                    "batch_size": 512,
                    "n_threads": 4,
                },
            },
        ],
    }
)

with np.load(data_paths[0], allow_pickle=False) as data:
    X = data["X"]
    y = data["y"]
    class_names = [str(value) for value in data["class_names"]]
    result = train_ensemble(
        X,
        y,
        groups=data["groups"],
        seeds=data["seeds"],
        seats=data["seats"],
        feature_names=[str(value) for value in data["feature_names"]],
        config=config,
    )

output = Path("/kaggle/working/ensemble_router_v4")
manifest = result.save(output)

def add_recalls(metrics, probabilities):
    values = recall_score(
        y,
        np.asarray(probabilities).argmax(axis=1),
        labels=np.arange(len(class_names)),
        average=None,
        zero_division=0,
    )
    metrics["per_class_recall"] = [float(value) for value in values]
    metrics["per_class_recall_named"] = {
        name: float(values[index]) for index, name in enumerate(class_names)
    }

for report in result.model_reports:
    if report.get("status") == "trained":
        add_recalls(
            report["metrics"]["calibrated_oof"],
            result.oof_probabilities[report["name"]],
        )
for key in ("blend_oof", "stack_oof", "hybrid_oof"):
    add_recalls(result.metrics[key], result.oof_probabilities[key.replace("_oof", "")])

unique, counts = np.unique(y, return_counts=True)
summary = {
    "schema_version": "kaggriculture-joint-router/v2",
    "elapsed_seconds": time.perf_counter() - started,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "rows": int(len(X)),
    "features": int(X.shape[1]),
    "class_counts": {class_names[int(index)]: int(count) for index, count in zip(unique, counts)},
    "all_25_actions_present": len(unique) == 25,
    "model_count": sum(report.get("status") == "trained" for report in result.model_reports),
    "model_reports": result.model_reports,
    "metrics": result.metrics,
    "split_audit": result.split_audit,
    "artifact": manifest.get("artifact"),
}
with (Path("/kaggle/working") / "training_summary_v4.json").open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps(summary, ensure_ascii=False), flush=True)
