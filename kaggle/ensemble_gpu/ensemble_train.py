"""Kaggle GPU job for the eight-model expert-router ensemble."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


os.environ.setdefault("PYTHONUNBUFFERED", "1")
started = time.perf_counter()

# Kaggle may allocate a Pascal P100 (sm_60).  Current default PyTorch wheels
# can omit Pascal kernels, so pin the last known-good CUDA 12.6 build first.
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
# Install the public repository, PyTabKit, and its FTT sklearn bridge.
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

from farmer_ensemble.config import EnsembleConfig
from farmer_ensemble.ensemble import train_ensemble


data_paths = list(Path("/kaggle/input").rglob("router_train.npz"))
if len(data_paths) != 1:
    raise RuntimeError(f"expected one router_train.npz, found {data_paths}")

config = EnsembleConfig.from_dict(
    {
        "n_splits": 4,
        "random_state": 20260825,
        "class_balance": "balanced",
        "unsupported_weight_strategy": "weighted_resample",
        "calibration": "temperature",
        "prediction": "hybrid",
        "hybrid_blend_weight": 0.5,
        "stacker_c": 1.0,
        "blend_iterations": 500,
        "blend_learning_rate": 0.04,
        "group_by_seed": True,
        "skip_failed_models": True,
        "model_size_budget_mib": 95.0,
        "models": [
            {
                "name": "lightgbm",
                "params": {
                    "n_estimators": 320,
                    "learning_rate": 0.04,
                    "num_leaves": 31,
                    "max_depth": 10,
                    "n_jobs": 4,
                },
            },
            {
                "name": "xgboost",
                "params": {
                    "n_estimators": 320,
                    "learning_rate": 0.04,
                    "max_depth": 8,
                    "tree_method": "hist",
                    "device": "cuda",
                    "n_jobs": 4,
                },
            },
            {
                "name": "catboost",
                "params": {
                    "iterations": 320,
                    "learning_rate": 0.04,
                    "depth": 8,
                    "task_type": "GPU",
                    "devices": "0",
                    "thread_count": 4,
                },
            },
            {"name": "hgbc", "params": {"max_iter": 320, "max_leaf_nodes": 31}},
            {
                "name": "extra_trees",
                "params": {
                    "n_estimators": 320,
                    "max_depth": 18,
                    "min_samples_leaf": 2,
                    "n_jobs": 4,
                },
            },
            {
                "name": "logistic_regression",
                "params": {"C": 1.0, "max_iter": 800},
            },
            {
                "name": "ft_transformer",
                "params": {
                    "device": "cuda",
                    "max_epochs": 48,
                    "batch_size": 256,
                    "n_threads": 4,
                },
            },
            {
                "name": "realmlp",
                "params": {
                    "device": "cuda",
                    "n_epochs": 48,
                    "batch_size": 256,
                    "n_threads": 4,
                },
            },
        ],
    }
)

with np.load(data_paths[0], allow_pickle=False) as data:
    result = train_ensemble(
        data["X"],
        data["y"],
        groups=data["groups"],
        seeds=data["seeds"],
        seats=data["seats"],
        feature_names=[str(value) for value in data["feature_names"]],
        config=config,
    )

output = Path("/kaggle/working/ensemble_router")
manifest = result.save(output)
summary = {
    "elapsed_seconds": time.perf_counter() - started,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "rows": 5752,
    "model_reports": result.model_reports,
    "metrics": result.metrics,
    "artifact": manifest.get("artifact"),
}
with (Path("/kaggle/working") / "training_summary.json").open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(json.dumps(summary, ensure_ascii=False), flush=True)
