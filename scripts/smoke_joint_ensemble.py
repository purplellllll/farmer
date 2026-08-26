"""Run a trained complete-action bundle in real official simulator games."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from farmer_ensemble.ensemble import load_bundle
from farmer_ensemble.joint_policy import JointEnsemblePolicy
from farmer_rl.actions import ANIMAL_COSTS, SEED_COSTS
from farmer_rl.environment import KaggricultureEnv, pass_action


def force_cpu(bundle) -> None:
    for item in bundle.base_models:
        estimator = item.adapter.estimator
        if item.name == "xgboost":
            estimator.set_params(device="cpu", n_jobs=1)
            estimator.get_booster().set_param({"device": "cpu", "nthread": 1})
        elif item.name in {"lightgbm", "extra_trees"}:
            try:
                estimator.set_params(n_jobs=1)
            except (TypeError, ValueError):
                pass
        if item.name in {"ft_transformer", "realmlp"}:
            estimator.device = "cpu"
            interface = getattr(estimator, "alg_interface_", None)
            if hasattr(interface, "device"):
                interface.device = "cpu"
            for sub_interface in getattr(interface, "sub_split_interfaces", []):
                if hasattr(sub_interface, "device"):
                    sub_interface.device = "cpu"
                model = getattr(sub_interface, "model", None)
                if model is not None and hasattr(model, "device"):
                    model.device = "cpu"


def conservative_market_cost(observation, orders) -> int:
    seat = int(observation["player"])
    farm = observation["farms"][seat]
    prices = (observation.get("market", {}) or {}).get("prices", {}) or {}
    hires = int(farm.get("hires_today", 0) or 0)
    unlocked = len(farm.get("unlocked_quadrants", []) or ["NW"])
    total = 0
    for order in orders:
        operation = str(order[0])
        if operation == "BUY_SEED":
            total += SEED_COSTS[str(order[1])] * int(order[2])
        elif operation == "BUY_ANIMAL":
            total += ANIMAL_COSTS[str(order[1])] * int(order[2])
        elif operation == "BUY_PRODUCT":
            total += int(np.ceil(1.1 * float(prices.get(str(order[1]), 0) or 0))) * int(order[2])
        elif operation == "HIRE":
            total += JointEnsemblePolicy._fib(hires)
            hires += 1
        elif operation == "BUY_LAND":
            total += (1000, 2000, 4000)[min(unlocked - 1, 2)]
            unlocked += 1
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--episode-steps", type=int, default=96)
    parser.add_argument("--seed-start", type=int, default=20261000)
    parser.add_argument("--opponent", choices=("starter", "pass"), default="starter")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    bundle = load_bundle(args.bundle)
    force_cpu(bundle)
    policy = JointEnsemblePolicy(bundle)
    env = KaggricultureEnv(configuration={"episodeSteps": args.episode_steps})
    episodes = []
    failures = []
    started = time.perf_counter()
    for episode in range(args.episodes):
        seed = args.seed_start + episode
        observations = env.reset(seed=seed)
        duplicate_orders = 0
        budget_violations = 0
        max_market_orders = 0
        for step in range(args.episode_steps):
            try:
                own_action = policy(observations[episode % 2])
                duplicate_orders += len(own_action["market"]) - len(
                    {tuple(order) for order in own_action["market"]}
                )
                max_market_orders = max(max_market_orders, len(own_action["market"]))
                own_money = int(
                    observations[episode % 2]["farms"][episode % 2].get("money", 0) or 0
                )
                budget_violations += int(
                    conservative_market_cost(observations[episode % 2], own_action["market"])
                    > own_money
                )
                opponent_seat = 1 - episode % 2
                opponent_farm = observations[opponent_seat]["farms"][opponent_seat]
                if args.opponent == "starter":
                    from kaggle_environments.envs.kaggriculture.kaggriculture import starter_agent

                    opponent_action = starter_agent(observations[opponent_seat])
                    opponent_action.setdefault(
                        "hands", [["PASS"] for _ in (opponent_farm.get("hands", []) or [])]
                    )
                else:
                    opponent_action = pass_action(len(opponent_farm.get("hands", []) or []))
                actions = {
                    episode % 2: own_action,
                    opponent_seat: opponent_action,
                }
                result = env.step(actions)
            except Exception as exc:
                failures.append(f"seed={seed} step={step}: {type(exc).__name__}: {exc}")
                break
            observations = result.observations
            if result.terminated:
                break
        raw = env.raw_environment.steps[-1]
        episodes.append(
            {
                "seed": seed,
                "model_seat": episode % 2,
                "rewards": [float(state.reward or 0.0) for state in raw],
                "statuses": [str(state.status) for state in raw],
                "duplicate_market_orders": duplicate_orders,
                "conservative_budget_violations": budget_violations,
                "max_market_orders": max_market_orders,
            }
        )
    summary = {
        "episodes": episodes,
        "opponent": args.opponent,
        "decode_action_counts": dict(sorted(policy.decode_stats.items())),
        "failures": failures,
        "elapsed_seconds": time.perf_counter() - started,
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
