"""Command-line entry for config validation, BC warm start and PPO self-play."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .bc import train_bc
from .model import ModelConfig
from .native_ppo import run_native_self_play
from .rllib_entry import run_self_play


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    model = ModelConfig.from_dict(dict(config.get("model", {})))
    if model.slots != 1 + int(config.get("actions", {}).get("max_hands", 16)) + 10:
        raise ValueError("model.slots must equal 1 + actions.max_hands + 10 market slots")
    if model.candidate_capacity != int(config.get("actions", {}).get("candidate_capacity", 64)):
        raise ValueError("model/action candidate_capacity values must match")
    return config


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate configuration without optional dependencies")
    validate.add_argument("--config", required=True)
    bc = subparsers.add_parser("bc", help="behaviour-clone licensed local trajectories")
    bc.add_argument("--config", required=True)
    bc.add_argument("--input", action="append", required=True, help="trajectory JSONL; repeatable")
    bc.add_argument("--output", required=True)
    bc.add_argument("--epochs", type=int, default=3)
    bc.add_argument("--batch-size", type=int, default=32)
    bc.add_argument("--device", default="cpu")
    bc.add_argument("--initial-checkpoint")
    ppo = subparsers.add_parser("self-play", help="RLlib PPO with frozen checkpoint slots")
    ppo.add_argument("--config", required=True)
    ppo.add_argument("--iterations", type=int, required=True)
    ppo.add_argument("--bc-checkpoint")
    native = subparsers.add_parser(
        "native-self-play", help="single-process CUDA PPO for Windows/local training"
    )
    native.add_argument("--config", required=True)
    native.add_argument("--iterations", type=int, required=True)
    native.add_argument("--output", required=True)
    native.add_argument("--resume")
    native.add_argument("--bc-checkpoint")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "validate":
        print(json.dumps({"ok": True, "config": args.config}, ensure_ascii=False))
        return 0
    if args.command == "bc":
        result = train_bc(
            input_paths=args.input,
            output_path=args.output,
            model_config=ModelConfig.from_dict(dict(config.get("model", {}))),
            epochs=args.epochs,
            learning_rate=float(config.get("bc", {}).get("lr", 3e-4)),
            device=args.device,
            batch_size=args.batch_size,
            initial_checkpoint=args.initial_checkpoint,
        )
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.command == "native-self-play":
        history = run_native_self_play(
            config,
            iterations=args.iterations,
            output_dir=args.output,
            resume=args.resume,
            bc_checkpoint=args.bc_checkpoint,
        )
        summary = history[-1] if history else {}
        print(json.dumps({"iterations": len(history), "last": summary}, ensure_ascii=False))
        return 0
    history = run_self_play(config, iterations=args.iterations, bc_checkpoint=args.bc_checkpoint)
    summary = history[-1] if history else {}
    print(json.dumps({"iterations": len(history), "last": summary}, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
