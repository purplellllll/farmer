from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import EnsembleConfig
from .ensemble import train_ensemble


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a leakage-aware Kaggriculture helper ensemble from an NPZ dataset."
    )
    parser.add_argument("--data", required=True, help="NPZ containing X, y, groups and metadata")
    parser.add_argument("--config", required=True, help="ensemble JSON configuration")
    parser.add_argument("--output", required=True, help="artifact output directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = EnsembleConfig.load(args.config)
    with np.load(args.data, allow_pickle=False) as data:
        required = {"X", "y", "groups"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"dataset is missing arrays: {sorted(missing)}")
        feature_names = (
            [str(value) for value in data["feature_names"]]
            if "feature_names" in data.files
            else None
        )
        result = train_ensemble(
            data["X"],
            data["y"],
            groups=data["groups"],
            seeds=data["seeds"] if "seeds" in data.files else None,
            seats=data["seats"] if "seats" in data.files else None,
            feature_names=feature_names,
            config=config,
        )
    manifest = result.save(Path(args.output))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0
