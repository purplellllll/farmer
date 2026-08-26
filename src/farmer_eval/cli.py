"""Command-line entry point for local strategy benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import BenchmarkConfig, run_benchmark
from .policies import BUILTIN_POLICIES


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paired-seed Kaggriculture strategy benchmark")
    parser.add_argument("--config", help="JSON config (CLI values override it)")
    parser.add_argument("--candidate", help="Policy spec to evaluate")
    parser.add_argument("--opponent", action="append", dest="opponents", help="Repeat for each opponent policy spec")
    parser.add_argument("--seeds", help="Comma-separated integer seeds")
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--seed-count", type=int)
    parser.add_argument("--episode-steps", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--timeout", type=float, dest="policy_timeout_seconds")
    parser.add_argument("--output-dir")
    parser.add_argument("--no-swap", action="store_true", help="Evaluate candidate only in seat 0")
    parser.add_argument("--list-policies", action="store_true")
    return parser


def _load(args: argparse.Namespace) -> BenchmarkConfig:
    payload: dict = {}
    if args.config:
        with Path(args.config).open(encoding="utf-8") as handle:
            payload = json.load(handle)
    if args.candidate:
        payload["candidate"] = args.candidate
    if args.opponents:
        payload["opponents"] = args.opponents
    if args.seeds:
        payload["seeds"] = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    elif args.seed_start is not None or args.seed_count is not None:
        payload.pop("seeds", None)
        payload["seed_start"] = args.seed_start if args.seed_start is not None else int(payload.get("seed_start", 0))
        payload["seed_count"] = args.seed_count if args.seed_count is not None else int(payload.get("seed_count", 1))
    for name in ("episode_steps", "workers", "policy_timeout_seconds", "output_dir"):
        value = getattr(args, name)
        if value is not None:
            payload[name] = value
    if args.no_swap:
        payload["swap_seats"] = False
    missing = [name for name in ("candidate", "opponents") if name not in payload]
    if missing:
        raise SystemExit(f"missing required config: {', '.join(missing)}")
    return BenchmarkConfig.from_dict(payload)


def main() -> None:
    args = _parser().parse_args()
    if args.list_policies:
        for item in BUILTIN_POLICIES.values():
            print(f"{item.name:12} {item.family:12} {item.description}")
        return
    config = _load(args)
    result = run_benchmark(config)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {Path(config.output_dir).resolve() / 'benchmark.json'}")


if __name__ == "__main__":
    main()
