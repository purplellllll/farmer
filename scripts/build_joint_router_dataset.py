"""Collect diverse legal self-play and build the complete v2 router table."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import random
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from farmer_ensemble.joint_router_dataset import DECISION_NAMES, build_joint_router_npz
from farmer_rl.actions import ANIMALS, CROPS, CandidateGenerator
from farmer_rl.environment import KaggricultureEnv


PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
SHED_TILES = {(4, 4), (5, 4), (4, 5), (5, 5)}
LAND_COSTS = (1000, 2000, 4000)
ANIMAL_STRUCTURE = {"GOOSE": "COOP", "COW": "PASTURE", "SHEEP": "PASTURE"}


def _inventory_total(inventories: list[Mapping[str, Any]], item: str) -> int:
    return sum(int(inv.get(item, 0) or 0) for inv in inventories if isinstance(inv, Mapping))


def _tile_at(board: list[list[Any]], position: list[int]) -> Any:
    x, y = (int(v) for v in position)
    return board[y][x]


def _nearest(position: list[int], targets: list[tuple[int, int]]) -> tuple[int, int] | None:
    if not targets:
        return None
    x, y = (int(v) for v in position)
    return min(targets, key=lambda point: (abs(point[0] - x) + abs(point[1] - y), point[1], point[0]))


def _move_toward(position: list[int], target: tuple[int, int], phase: int) -> list[str]:
    x, y = (int(v) for v in position)
    tx, ty = target
    horizontal = ["EAST"] if tx > x else (["WEST"] if tx < x else [])
    vertical = ["SOUTH"] if ty > y else (["NORTH"] if ty < y else [])
    choices = (horizontal, vertical) if phase % 2 == 0 else (vertical, horizontal)
    for choice in choices:
        if choice:
            return choice
    return ["PASS"]


class DiverseCurriculumPolicy:
    """State-only curriculum that exercises every legal action family.

    This is data generation, not a leaderboard claim.  It mixes profitable crop
    loops with expansion, animal care, logistics, and bounded exploration.  All
    submitted market orders are unique and are budgeted sequentially.
    """

    def __init__(self, *, seed: int, profile: int) -> None:
        self.rng = random.Random(seed)
        self.profile = int(profile)
        self.counts: Counter[str] = Counter()

    def _market(self, obs: Mapping[str, Any]) -> list[list[Any]]:
        seat = int(obs["player"])
        farm = obs["farms"][seat]
        private = obs.get("private", {}) or {}
        shed = private.get("shed", {}) or {}
        seeds = private.get("seeds", {}) or {}
        inventories = private.get("inventories", []) or []
        prices = (obs.get("market", {}) or {}).get("prices", {}) or {}
        money = int(farm.get("money", 0) or 0)
        budget = money
        orders: list[list[Any]] = []
        seen: set[tuple[Any, ...]] = set()

        def add(order: list[Any], cost: int = 0) -> bool:
            nonlocal budget
            key = tuple(order)
            if key in seen or len(orders) >= 10 or cost > budget:
                return False
            seen.add(key)
            orders.append(order)
            budget -= max(0, int(cost))
            return True

        day, hour = int(obs.get("day", 0)), int(obs.get("hour", 0))
        unlocked = len(farm.get("unlocked_quadrants", []) or ["NW"])
        if unlocked < 4 and (day == 0 or day in (6, 14)):
            add(["BUY_LAND"], LAND_COSTS[min(unlocked - 1, 2)])
        # A bounded daily hand count creates genuine hand-policy examples without
        # the ten identical HIRE orders emitted by the failed independent decoder.
        if hour == 0 and len(farm.get("hands", []) or []) < 2:
            add(["HIRE"], 1 << min(int(farm.get("hires_today", 0) or 0), 6))

        crop = CROPS[(day + self.profile) % len(CROPS)]
        if int(seeds.get(crop, 0) or 0) < 3:
            seed_cost = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}[crop]
            add(["BUY_SEED", crop, 3], 3 * seed_cost)

        # Purchase each animal at most once per joint action and only while the
        # farm does not already own one in shed, inventory, or a structure.
        board = farm.get("tiles", []) or []
        animals_on_board = {
            str(tile.get("animal"))
            for row in board
            for tile in row
            if isinstance(tile, Mapping) and tile.get("animal")
        }
        animal = ANIMALS[(day // 3 + self.profile) % len(ANIMALS)]
        owned = int(shed.get(animal, 0) or 0) + _inventory_total(inventories, animal)
        if day <= 12 and owned == 0 and animal not in animals_on_board:
            add(["BUY_ANIMAL", animal, 1], {"GOOSE": 300, "COW": 400, "SHEEP": 500}[animal])

        # BUY_PRODUCT is required for animal feed and fertilizer interactions.
        wheat_owned = int(shed.get("WHEAT", 0) or 0) + _inventory_total(inventories, "WHEAT")
        if wheat_owned < 4 and day <= 20:
            add(["BUY_PRODUCT", "WHEAT", 4], 4 * int(prices.get("WHEAT", 25) or 25))
        fertilizer_owned = int(shed.get("FERTILIZER", 0) or 0) + _inventory_total(inventories, "FERTILIZER")
        if fertilizer_owned == 0 and day in (2, 8, 16):
            add(["BUY_PRODUCT", "FERTILIZER", 1], int(prices.get("FERTILIZER", 100) or 100))

        # One order per product, descending price, retains ordered-market variety.
        sale_candidates = [
            (int(prices.get(item, 0) or 0), item, int(shed.get(item, 0) or 0))
            for item in PRODUCTS
            if int(shed.get(item, 0) or 0) > 0
        ]
        for _, item, quantity in sorted(sale_candidates, reverse=True)[:3]:
            add(["SELL", item, quantity])
        return orders

    def _unit_action(self, obs: Mapping[str, Any], unit_index: int) -> list[Any]:
        seat = int(obs["player"])
        farm = obs["farms"][seat]
        private = obs.get("private", {}) or {}
        positions = [farm.get("farmer", [0, 0]), *(farm.get("hands", []) or [])]
        position = positions[unit_index]
        x, y = (int(v) for v in position)
        board = farm.get("tiles", []) or []
        tile = _tile_at(board, position)
        inventories = private.get("inventories", []) or []
        inventory = inventories[unit_index] if unit_index < len(inventories) else {}
        inventory = inventory if isinstance(inventory, Mapping) else {}
        shed = private.get("shed", {}) or {}
        seeds = private.get("seeds", {}) or {}
        day = int(obs.get("day", 0))
        phase = int(obs.get("step", 0)) + unit_index + self.profile

        # Execute high-value state-local actions before navigation.
        if isinstance(tile, Mapping):
            kind = str(tile.get("kind", ""))
            if kind == "WEED":
                return ["DIG"]
            if kind == "PLANT":
                age = day - int(tile.get("planted_day", day) or day)
                if int(tile.get("yield_units", 0) or 0) > 0 and (age >= 3 or phase % 5 == 0):
                    return ["HARVEST"]
                if int(inventory.get("FERTILIZER", 0) or 0) > 0 and int(tile.get("fertilized_until_day", -1) or -1) < day:
                    return ["FERTILIZE"]
                if not bool(tile.get("watered_today", False)):
                    return ["WATER"]
                if age > 18 and phase % 11 == 0:
                    return ["DIG"]
            if kind in {"COOP", "PASTURE"}:
                animal = tile.get("animal")
                if animal:
                    if not bool(tile.get("fed_today", False)) and int(inventory.get("WHEAT", 0) or 0) > 0:
                        return ["FEED"]
                    if not bool(tile.get("cared_today", False)):
                        return ["CARE"]
                    if int(tile.get("yield_units", 0) or 0) > 0:
                        return ["HARVEST"]
                    if bool(tile.get("fertilizer_available", False)):
                        return ["COLLECT_FERTILIZER"]
                else:
                    for animal_name in ANIMALS:
                        if ANIMAL_STRUCTURE[animal_name] == kind and int(inventory.get(animal_name, 0) or 0) > 0:
                            return ["PLACE", animal_name]

        at_shed = (x, y) in SHED_TILES
        carried_products = sum(int(inventory.get(item, 0) or 0) for item in PRODUCTS)
        if at_shed:
            if carried_products > 3 or (carried_products > 0 and phase % 7 == 0):
                return ["DROP"]
            for item in (*ANIMALS, "FERTILIZER", "WHEAT"):
                if int(shed.get(item, 0) or 0) > 0 and int(inventory.get(item, 0) or 0) == 0:
                    return ["PICKUP", item, 1 if item != "WHEAT" else min(6, int(shed.get(item, 0) or 0))]

        # Carrying resources gives navigation a purposeful destination.
        if any(int(inventory.get(animal, 0) or 0) > 0 for animal in ANIMALS):
            targets = []
            for yy, row in enumerate(board):
                for xx, candidate in enumerate(row):
                    if not isinstance(candidate, Mapping) or candidate.get("animal"):
                        continue
                    if any(
                        int(inventory.get(animal, 0) or 0) > 0
                        and candidate.get("kind") == ANIMAL_STRUCTURE[animal]
                        for animal in ANIMALS
                    ):
                        targets.append((xx, yy))
            target = _nearest(position, targets)
            if target is not None:
                return _move_toward(position, target, phase)
        if int(inventory.get("FERTILIZER", 0) or 0) > 0:
            targets = [
                (xx, yy)
                for yy, row in enumerate(board)
                for xx, candidate in enumerate(row)
                if isinstance(candidate, Mapping) and candidate.get("kind") == "PLANT"
            ]
            target = _nearest(position, targets)
            if target is not None:
                return _move_toward(position, target, phase)
        if int(inventory.get("WHEAT", 0) or 0) > 0:
            targets = [
                (xx, yy)
                for yy, row in enumerate(board)
                for xx, candidate in enumerate(row)
                if isinstance(candidate, Mapping)
                and candidate.get("animal")
                and not candidate.get("fed_today", False)
            ]
            target = _nearest(position, targets)
            if target is not None:
                return _move_toward(position, target, phase)

        # Empty tiles are split among crops and animal structures.  Building is
        # throttled so the farm remains productive rather than becoming a maze.
        if tile is None:
            structure_count = sum(
                isinstance(candidate, Mapping) and candidate.get("kind") in {"COOP", "PASTURE"}
                for row in board
                for candidate in row
            )
            if structure_count < 4 and phase % 13 in (0, 1):
                return ["BUILD_COOP"] if structure_count % 3 == 0 else ["BUILD_PASTURE"]
            available = [crop for crop in CROPS if int(seeds.get(crop, 0) or 0) > 0]
            if available:
                crop = available[(phase + self.profile) % len(available)]
                return ["PLANT", crop]

        # Seek actionable tiles, then shed, then explore with all four moves.
        targets = [
            (xx, yy)
            for yy, row in enumerate(board)
            for xx, candidate in enumerate(row)
            if isinstance(candidate, Mapping) and candidate.get("kind") in {"PLANT", "WEED", "COOP", "PASTURE"}
        ]
        target = _nearest(position, targets)
        if target is not None and target != (x, y) and phase % 4 != 0:
            return _move_toward(position, target, phase)
        if carried_products > 0 or any(int(inventory.get(item, 0) or 0) > 0 for item in ANIMALS):
            target = _nearest(position, list(SHED_TILES))
            if target is not None and target != (x, y):
                return _move_toward(position, target, phase)
        moves = ("NORTH", "EAST", "SOUTH", "WEST", "PASS")
        for offset in range(len(moves)):
            operation = moves[(phase + offset) % len(moves)]
            if operation == "NORTH" and y > 0:
                return [operation]
            if operation == "SOUTH" and y + 1 < len(board):
                return [operation]
            if operation == "WEST" and x > 0:
                return [operation]
            if operation == "EAST" and x + 1 < len(board[0]):
                return [operation]
            if operation == "PASS":
                return [operation]
        return ["PASS"]

    def __call__(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        seat = int(obs["player"])
        hand_count = len(obs["farms"][seat].get("hands", []) or [])
        unit_actions = [self._unit_action(obs, index) for index in range(hand_count + 1)]
        market = self._market(obs)
        for action in (*unit_actions, *market):
            self.counts[str(action[0])] += 1
        return {"farmer": unit_actions[0], "hands": unit_actions[1:], "market": market}


def collect(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "curriculum_v2.jsonl"
    env = KaggricultureEnv(configuration={"episodeSteps": args.episode_steps})
    policies: list[DiverseCurriculumPolicy] = []
    records = 0
    with raw_path.open("w", encoding="utf-8") as handle:
        for episode_index in range(args.episodes):
            seed = args.seed_start + episode_index
            episode_id = f"curriculum-v2-{seed}"
            seat_policies = {
                seat: DiverseCurriculumPolicy(seed=seed * 17 + seat, profile=(episode_index + seat * 2) % 5)
                for seat in (0, 1)
            }
            policies.extend(seat_policies.values())
            observations = env.reset(seed=seed)
            for step in range(args.episode_steps):
                actions = {seat: seat_policies[seat](deepcopy(observations[seat])) for seat in (0, 1)}
                result = env.step(actions)
                for seat in (0, 1):
                    record = {
                        "episode_id": episode_id,
                        "step": step,
                        "acting_seat": seat,
                        "observation": observations[seat],
                        "action": actions[seat],
                        "seed": seed,
                        "policy_id": f"diverse-curriculum-p{seat}",
                        "source_id": "self-generated-official-simulator-1.32.7",
                    }
                    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    records += 1
                observations = result.observations
                if result.terminated:
                    break
            print(f"collected {episode_id}", flush=True)
    npz_path = args.output_dir / "joint_router_train.npz"
    summary = build_joint_router_npz([raw_path], npz_path)
    summary.update(
        episodes=args.episodes,
        raw_records=records,
        seed_start=args.seed_start,
        episode_steps=args.episode_steps,
        policy_action_counts=dict(sorted(sum((policy.counts for policy in policies), Counter()).items())),
        source="self-generated with kaggle-environments==1.32.7",
    )
    with (args.output_dir / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/ensemble-retrain-v2/data"))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument("--seed-start", type=int, default=20260900)
    return parser.parse_args()


if __name__ == "__main__":
    collect(parse_args())
