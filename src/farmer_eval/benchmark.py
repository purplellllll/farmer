"""Paired-seed, swapped-seat local benchmark runner."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict, dataclass
import csv
import json
import math
from pathlib import Path
import queue
import statistics
import threading
import time
from typing import Any, Callable, Mapping

from farmer_rl.environment import KaggricultureEnv, pass_action, validate_action
from farmer_rl.actions import ANIMAL_COSTS, SEED_COSTS

from .policies import Policy, load_policy


@dataclass(frozen=True)
class BenchmarkConfig:
    candidate: str
    opponents: tuple[str, ...]
    seeds: tuple[int, ...]
    episode_steps: int = 720
    swap_seats: bool = True
    policy_timeout_seconds: float = 1.0
    max_policy_faults: int = 3
    workers: int = 1
    output_dir: str = "artifacts/eval/latest"
    debug: bool = False

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BenchmarkConfig":
        seeds = value.get("seeds")
        if seeds is None:
            start = int(value.get("seed_start", 0))
            seeds = range(start, start + int(value.get("seed_count", 1)))
        return cls(
            candidate=str(value["candidate"]),
            opponents=tuple(str(item) for item in value["opponents"]),
            seeds=tuple(int(item) for item in seeds),
            episode_steps=int(value.get("episode_steps", 720)),
            swap_seats=bool(value.get("swap_seats", True)),
            policy_timeout_seconds=float(value.get("policy_timeout_seconds", 1.0)),
            max_policy_faults=int(value.get("max_policy_faults", 3)),
            workers=int(value.get("workers", 1)),
            output_dir=str(value.get("output_dir", "artifacts/eval/latest")),
            debug=bool(value.get("debug", False)),
        )


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] * (upper - index) + ordered[upper] * (index - lower))


def _hand_count(observation: Mapping[str, Any]) -> int:
    seat = int(observation.get("player", -1))
    farms = observation.get("farms", [])
    if seat not in (0, 1) or not isinstance(farms, list) or len(farms) != 2:
        return 0
    return len(farms[seat].get("hands", []) or [])


def _operation_counts(action: Mapping[str, Any]) -> Counter[str]:
    operations: Counter[str] = Counter()
    farmer = action.get("farmer", ["PASS"]) or ["PASS"]
    operations[str(farmer[0])] += 1
    for item in action.get("hands", []) or []:
        operations[str((item or ["PASS"])[0])] += 1
    for item in action.get("market", []) or []:
        operations[str((item or ["NO_ORDER"])[0])] += 1
    return operations


class SafePolicy:
    """Bound a policy call, normalize faults to PASS, and retain diagnostics.

    The policy function runs on a daemon thread so a Python-level hang cannot
    block the match.  After a timeout it is disabled for the rest of that match.
    Native extensions that hold the GIL cannot be safely pre-empted in-process;
    use ``workers>1`` for whole-match process isolation when evaluating unknown
    third-party agents.
    """

    def __init__(self, policy: Policy, *, timeout: float, max_faults: int) -> None:
        self.policy = policy
        self.timeout = float(timeout)
        self.max_faults = max(1, int(max_faults))
        self.calls = 0
        self.exceptions = 0
        self.timeouts = 0
        self.invalid_actions = 0
        self.disabled_calls = 0
        self.latencies: list[float] = []
        self.actions: Counter[str] = Counter()
        self.joint_action_signatures: Counter[str] = Counter()
        self.market_orders = 0
        self.duplicate_market_orders = 0
        self.resource_conflict_orders = 0
        self.overbudget_orders = 0
        self.pass_streak = 0
        self.max_pass_streak = 0
        self.money_trace: list[tuple[int, float]] = []
        self.disabled = False

    def _record_observation(self, observation: Mapping[str, Any]) -> None:
        seat = int(observation.get("player", -1))
        farms = observation.get("farms", [])
        if seat in (0, 1) and isinstance(farms, list) and len(farms) == 2:
            money = float(farms[seat].get("money", 0.0) or 0.0)
            self.money_trace.append((int(observation.get("step", self.calls - 1) or 0), money))

    def _record_action(self, action: Mapping[str, Any], observation: Mapping[str, Any]) -> None:
        self.actions.update(_operation_counts(action))
        signature = json.dumps(action, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        self.joint_action_signatures[signature] += 1
        farmer = action.get("farmer", ["PASS"]) or ["PASS"]
        hands = action.get("hands", []) or []
        market = action.get("market", []) or []
        idle = str(farmer[0]) == "PASS" and all(str((item or ["PASS"])[0]) == "PASS" for item in hands) and not market
        self.pass_streak = self.pass_streak + 1 if idle else 0
        self.max_pass_streak = max(self.max_pass_streak, self.pass_streak)

        market_signatures = [json.dumps(item, ensure_ascii=True, separators=(",", ":")) for item in market]
        self.market_orders += len(market_signatures)
        self.duplicate_market_orders += len(market_signatures) - len(set(market_signatures))
        seat = int(observation.get("player", -1))
        farms = observation.get("farms", [])
        private = observation.get("private", {}) or {}
        if seat not in (0, 1) or not isinstance(farms, list) or len(farms) != 2:
            return
        available = {
            str(key): int(value or 0)
            for key, value in ((private.get("shed", {}) or {}).items())
        }
        budget = int(float(farms[seat].get("money", 0) or 0))
        unlocked = len(farms[seat].get("unlocked_quadrants", []) or ["NW"])
        hires = int(farms[seat].get("hires_today", 0) or 0)

        def fib(index: int) -> int:
            a, b = 1, 1
            for _ in range(max(0, index)):
                a, b = b, a + b
            return a

        for item in market:
            if not item:
                continue
            op = str(item[0])
            cost = 0
            if op == "SELL" and len(item) >= 3:
                product, quantity = str(item[1]), int(item[2])
                if quantity > available.get(product, 0):
                    self.resource_conflict_orders += 1
                available[product] = available.get(product, 0) - quantity
            elif op == "BUY_SEED" and len(item) >= 3:
                cost = SEED_COSTS.get(str(item[1]), 0) * int(item[2])
            elif op == "BUY_ANIMAL" and len(item) >= 3:
                cost = ANIMAL_COSTS.get(str(item[1]), 0) * int(item[2])
            elif op == "HIRE":
                cost = fib(hires)
                hires += 1
            elif op == "BUY_LAND":
                if unlocked >= 4:
                    self.resource_conflict_orders += 1
                    continue
                cost = (1000, 2000, 4000)[min(unlocked - 1, 2)]
                unlocked += 1
            if cost > budget:
                self.overbudget_orders += 1
            else:
                budget -= cost

    def _fallback(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        action = pass_action(_hand_count(observation))
        self._record_action(action, observation)
        return action

    def act(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        self.calls += 1
        self._record_observation(observation)
        if self.disabled:
            self.disabled_calls += 1
            return self._fallback(observation)
        result_queue: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

        def invoke() -> None:
            try:
                result_queue.put((True, self.policy(deepcopy(dict(observation)))))
            except BaseException as exc:  # policy boundary must contain third-party failures
                result_queue.put((False, exc))

        started = time.perf_counter()
        worker = threading.Thread(target=invoke, daemon=True)
        worker.start()
        worker.join(self.timeout)
        elapsed = time.perf_counter() - started
        self.latencies.append(elapsed)
        if worker.is_alive():
            self.timeouts += 1
            self.disabled = True
            return self._fallback(observation)
        ok, value = result_queue.get_nowait()
        if not ok:
            self.exceptions += 1
            if self.exceptions + self.invalid_actions >= self.max_faults:
                self.disabled = True
            return self._fallback(observation)
        try:
            action = validate_action(value, hand_count=_hand_count(observation))
        except Exception:
            self.invalid_actions += 1
            if self.exceptions + self.invalid_actions >= self.max_faults:
                self.disabled = True
            return self._fallback(observation)
        self._record_action(action, observation)
        return action

    @property
    def faulted(self) -> bool:
        return bool(self.timeouts or self.exceptions or self.invalid_actions)

    def metrics(self) -> dict[str, Any]:
        operation_total = sum(self.actions.values())
        entropy = 0.0
        if operation_total:
            for count in self.actions.values():
                probability = count / operation_total
                entropy -= probability * math.log(probability)
        early = [money for step, money in self.money_trace if step <= 72]
        all_money = [money for _, money in self.money_trace]
        initial_money = all_money[0] if all_money else None
        early_minimum = min(early) if early else initial_money
        return {
            "calls": self.calls,
            "exceptions": self.exceptions,
            "timeouts": self.timeouts,
            "invalid_actions": self.invalid_actions,
            "disabled_calls": self.disabled_calls,
            "faulted": self.faulted,
            "latency_p50_ms": 1000.0 * _percentile(self.latencies, 0.50),
            "latency_p95_ms": 1000.0 * _percentile(self.latencies, 0.95),
            "latency_max_ms": 1000.0 * max(self.latencies, default=0.0),
            "action_distribution": dict(sorted(self.actions.items())),
            "operation_type_count": len(self.actions),
            "operation_entropy": entropy,
            "unique_joint_actions": len(self.joint_action_signatures),
            "unique_joint_action_rate": len(self.joint_action_signatures) / self.calls if self.calls else 0.0,
            "market_orders": self.market_orders,
            "duplicate_market_orders": self.duplicate_market_orders,
            "duplicate_market_order_rate": self.duplicate_market_orders / self.market_orders if self.market_orders else 0.0,
            "resource_conflict_orders": self.resource_conflict_orders,
            "overbudget_orders": self.overbudget_orders,
            "max_idle_pass_streak": self.max_pass_streak,
            "initial_money": initial_money,
            "early_minimum_money": early_minimum,
            "early_money_spent": max(0.0, initial_money - early_minimum) if initial_money is not None and early_minimum is not None else None,
            "minimum_observed_money": min(all_money) if all_money else None,
            "final_observed_money": all_money[-1] if all_money else None,
            "_latencies": self.latencies,
        }


@dataclass(frozen=True)
class MatchTask:
    candidate: str
    opponent: str
    seed: int
    candidate_seat: int
    episode_steps: int
    policy_timeout_seconds: float
    max_policy_faults: int
    debug: bool


def _approximate_rewards(observations: Mapping[int, Mapping[str, Any]]) -> dict[int, float]:
    result: dict[int, float] = {}
    for seat in (0, 1):
        farms = observations.get(seat, {}).get("farms", [])
        result[seat] = float(farms[seat].get("money", 0.0) or 0.0) if len(farms) == 2 else 0.0
    return result


def run_match(
    task: MatchTask,
    *,
    env_factory: Callable[[], KaggricultureEnv] | None = None,
) -> dict[str, Any]:
    candidate_policy = SafePolicy(
        load_policy(task.candidate, seed=task.seed * 2 + task.candidate_seat),
        timeout=task.policy_timeout_seconds,
        max_faults=task.max_policy_faults,
    )
    opponent_policy = SafePolicy(
        load_policy(task.opponent, seed=task.seed * 2 + (1 - task.candidate_seat)),
        timeout=task.policy_timeout_seconds,
        max_faults=task.max_policy_faults,
    )
    by_seat = {
        task.candidate_seat: candidate_policy,
        1 - task.candidate_seat: opponent_policy,
    }
    env = env_factory() if env_factory else KaggricultureEnv(
        configuration={"episodeSteps": task.episode_steps}, debug=task.debug
    )
    observations = env.reset(seed=task.seed)
    statuses = {0: "ACTIVE", 1: "ACTIVE"}
    rewards = {0: 0.0, 1: 0.0}
    engine_error: str | None = None
    steps = 0
    started = time.perf_counter()
    for _ in range(task.episode_steps):
        try:
            result = env.step({seat: by_seat[seat].act(observations[seat]) for seat in (0, 1)})
        except Exception as exc:
            engine_error = f"{type(exc).__name__}: {exc}"
            statuses = {0: "ERROR", 1: "ERROR"}
            rewards = _approximate_rewards(observations)
            break
        observations = result.observations
        rewards = result.rewards
        statuses = result.statuses
        steps += 1
        if result.terminated or result.truncated:
            break
    elapsed = time.perf_counter() - started
    candidate_reward = float(rewards[task.candidate_seat])
    opponent_reward = float(rewards[1 - task.candidate_seat])
    difference = candidate_reward - opponent_reward
    outcome = 1.0 if difference > 0 else 0.0 if difference < 0 else 0.5
    candidate_metrics = candidate_policy.metrics()
    opponent_metrics = opponent_policy.metrics()
    return {
        "candidate": task.candidate,
        "opponent": task.opponent,
        "seed": task.seed,
        "candidate_seat": task.candidate_seat,
        "candidate_reward": candidate_reward,
        "opponent_reward": opponent_reward,
        "score_difference": difference,
        "outcome": outcome,
        "steps": steps,
        "statuses": statuses,
        "wall_seconds": elapsed,
        "engine_error": engine_error,
        "candidate_policy": candidate_metrics,
        "opponent_policy": opponent_metrics,
    }


def _tasks(config: BenchmarkConfig) -> list[MatchTask]:
    if not config.opponents:
        raise ValueError("at least one opponent is required")
    if not config.seeds:
        raise ValueError("at least one seed is required")
    if config.episode_steps <= 1 or config.workers <= 0 or config.policy_timeout_seconds <= 0:
        raise ValueError("episode_steps, workers, and policy_timeout_seconds must be positive")
    seats = (0, 1) if config.swap_seats else (0,)
    return [
        MatchTask(
            candidate=config.candidate,
            opponent=opponent,
            seed=seed,
            candidate_seat=seat,
            episode_steps=config.episode_steps,
            policy_timeout_seconds=config.policy_timeout_seconds,
            max_policy_faults=config.max_policy_faults,
            debug=config.debug,
        )
        for opponent in config.opponents
        for seed in config.seeds
        for seat in seats
    ]


def _group_summary(games: list[Mapping[str, Any]]) -> dict[str, Any]:
    wins = sum(float(game["outcome"]) == 1.0 for game in games)
    draws = sum(float(game["outcome"]) == 0.5 for game in games)
    losses = len(games) - wins - draws
    score_differences = [float(game["score_difference"]) for game in games]
    latencies = [
        float(item)
        for game in games
        for item in game["candidate_policy"].get("_latencies", [])
    ]
    action_counts: Counter[str] = Counter()
    for game in games:
        action_counts.update(game["candidate_policy"].get("action_distribution", {}))
    faulted_games = sum(bool(game["candidate_policy"].get("faulted")) for game in games)
    engine_failures = sum(game.get("engine_error") is not None for game in games)
    market_orders = sum(int(game["candidate_policy"].get("market_orders", 0)) for game in games)
    duplicate_orders = sum(int(game["candidate_policy"].get("duplicate_market_orders", 0)) for game in games)
    early_spend = [
        float(game["candidate_policy"]["early_money_spent"])
        for game in games
        if game["candidate_policy"].get("early_money_spent") is not None
    ]
    return {
        "games": len(games),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / len(games) if games else 0.0,
        "draw_rate": draws / len(games) if games else 0.0,
        "mean_outcome": statistics.fmean(float(game["outcome"]) for game in games) if games else 0.0,
        "mean_score_difference": statistics.fmean(score_differences) if games else 0.0,
        "median_score_difference": statistics.median(score_differences) if games else 0.0,
        "policy_latency_p50_ms": 1000.0 * _percentile(latencies, 0.50),
        "policy_latency_p95_ms": 1000.0 * _percentile(latencies, 0.95),
        "policy_crash_rate": faulted_games / len(games) if games else 0.0,
        "engine_failure_rate": engine_failures / len(games) if games else 0.0,
        "action_distribution": dict(sorted(action_counts.items())),
        "operation_type_count": len(action_counts),
        "duplicate_market_orders": duplicate_orders,
        "duplicate_market_order_rate": duplicate_orders / market_orders if market_orders else 0.0,
        "resource_conflict_orders": sum(int(game["candidate_policy"].get("resource_conflict_orders", 0)) for game in games),
        "overbudget_orders": sum(int(game["candidate_policy"].get("overbudget_orders", 0)) for game in games),
        "max_idle_pass_streak": max((int(game["candidate_policy"].get("max_idle_pass_streak", 0)) for game in games), default=0),
        "mean_unique_joint_action_rate": statistics.fmean(float(game["candidate_policy"].get("unique_joint_action_rate", 0.0)) for game in games) if games else 0.0,
        "mean_early_money_spent": statistics.fmean(early_spend) if early_spend else None,
    }


def summarize_games(games: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_opponent: dict[str, Any] = {}
    by_seat: dict[str, Any] = {}
    for opponent in sorted({str(game["opponent"]) for game in games}):
        by_opponent[opponent] = _group_summary([game for game in games if game["opponent"] == opponent])
    for seat in (0, 1):
        selected = [game for game in games if int(game["candidate_seat"]) == seat]
        if selected:
            by_seat[str(seat)] = _group_summary(selected)
    paired: list[float] = []
    pairs: dict[tuple[str, int], list[float]] = {}
    for game in games:
        pairs.setdefault((str(game["opponent"]), int(game["seed"])), []).append(float(game["outcome"]))
    for outcomes in pairs.values():
        if len(outcomes) == 2:
            paired.append(statistics.fmean(outcomes))
    return {
        "overall": _group_summary(games),
        "by_opponent": by_opponent,
        "by_candidate_seat": by_seat,
        "paired_seed_groups": len(paired),
        "paired_seed_mean_outcome": statistics.fmean(paired) if paired else None,
    }


def _public_game(game: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(dict(game))
    value["candidate_policy"].pop("_latencies", None)
    value["opponent_policy"].pop("_latencies", None)
    return value


def _write_outputs(config: BenchmarkConfig, games: list[dict[str, Any]], summary: Mapping[str, Any]) -> None:
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "farmer-local-eval/v1",
        "config": asdict(config),
        "summary": summary,
        "games": [_public_game(game) for game in games],
    }
    with (output / "benchmark.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    fields = [
        "candidate", "opponent", "seed", "candidate_seat", "candidate_reward",
        "opponent_reward", "score_difference", "outcome", "steps", "wall_seconds",
        "engine_error", "candidate_faulted", "candidate_timeouts",
        "candidate_exceptions", "candidate_invalid_actions",
        "duplicate_market_order_rate", "resource_conflict_orders", "overbudget_orders",
        "max_idle_pass_streak", "unique_joint_action_rate", "early_money_spent",
    ]
    with (output / "games.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for game in games:
            writer.writerow({
                **{field: game.get(field) for field in fields},
                "candidate_faulted": game["candidate_policy"]["faulted"],
                "candidate_timeouts": game["candidate_policy"]["timeouts"],
                "candidate_exceptions": game["candidate_policy"]["exceptions"],
                "candidate_invalid_actions": game["candidate_policy"]["invalid_actions"],
                "duplicate_market_order_rate": game["candidate_policy"]["duplicate_market_order_rate"],
                "resource_conflict_orders": game["candidate_policy"]["resource_conflict_orders"],
                "overbudget_orders": game["candidate_policy"]["overbudget_orders"],
                "max_idle_pass_streak": game["candidate_policy"]["max_idle_pass_streak"],
                "unique_joint_action_rate": game["candidate_policy"]["unique_joint_action_rate"],
                "early_money_spent": game["candidate_policy"]["early_money_spent"],
            })


def run_benchmark(config: BenchmarkConfig, *, write_outputs: bool = True) -> dict[str, Any]:
    tasks = _tasks(config)
    if config.workers == 1:
        games = [run_match(task) for task in tasks]
    else:
        games = []
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            futures = [executor.submit(run_match, task) for task in tasks]
            for future in as_completed(futures):
                games.append(future.result())
        games.sort(key=lambda game: (str(game["opponent"]), int(game["seed"]), int(game["candidate_seat"])))
    summary = summarize_games(games)
    if write_outputs:
        _write_outputs(config, games, summary)
    return {"summary": summary, "games": [_public_game(game) for game in games]}
