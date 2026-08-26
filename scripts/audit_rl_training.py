"""Create a machine-readable audit of an RL run, checkpoint and benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


def _json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(path: Path, window: int) -> dict[str, Any]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        return {"iterations": 0}
    recent = rows[-window:]
    nonzero = [row for row in rows if float(row.get("learner_win_rate", 0.0) or 0.0) > 0]
    return {
        "iterations": len(rows),
        "last_iteration": int(rows[-1]["iteration"]),
        "promotions": sum(bool(row.get("promoted")) for row in rows),
        "last_nonzero_win_iteration": int(nonzero[-1]["iteration"]) if nonzero else None,
        "recent_window": len(recent),
        "recent_mean_win_rate": mean(float(row.get("learner_win_rate", 0.0) or 0.0) for row in recent),
        "recent_mean_score_difference": mean(float(row.get("mean_score_difference", 0.0) or 0.0) for row in recent),
        "recent_mean_seconds": mean(float(row.get("seconds", 0.0) or 0.0) for row in recent),
        "non_finite_numeric_values": sum(
            1
            for row in rows
            for value in row.values()
            if isinstance(value, float) and not math.isfinite(value)
        ),
    }


def _checkpoint(run_dir: Path) -> dict[str, Any]:
    import torch

    latest = _json(run_dir / "latest.json")
    path = Path(latest["checkpoint"])
    if not path.is_absolute():
        path = run_dir / path
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model", payload.get("state_dict", {}))
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "format": payload.get("format"),
        "iteration": int(payload.get("iteration", latest.get("iteration", 0))),
        "model_config": payload.get("model_config"),
        "parameter_count": sum(int(tensor.numel()) for tensor in state.values()),
        "non_finite_parameters": sum(int((~torch.isfinite(tensor)).sum().item()) for tensor in state.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    metrics = _metrics(args.run_dir / "metrics.jsonl", max(1, args.window))
    benchmark = _json(args.benchmark)["summary"]["overall"] if args.benchmark else None
    findings = []
    if metrics.get("recent_mean_win_rate") == 0:
        findings.append("zero_recent_win_rate")
    if metrics.get("promotions", 0) <= 1:
        findings.append("no_sustained_snapshot_promotion")
    if benchmark and float(benchmark.get("duplicate_market_order_rate", 0.0)) > 0:
        findings.append("duplicate_market_orders")
    if benchmark and int(benchmark.get("max_idle_pass_streak", 0)) >= 12:
        findings.append("long_pass_streak")
    result = {
        "schema_version": "farmer-rl-audit/v1",
        "run_dir": str(args.run_dir.resolve()),
        "metrics": metrics,
        "checkpoint": _checkpoint(args.run_dir),
        "benchmark": benchmark,
        "findings": findings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
