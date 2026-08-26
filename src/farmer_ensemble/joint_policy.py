"""Legality-masked decoder for a complete-action decision-router bundle."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

import numpy as np

from farmer_rl.actions import (
    ANIMALS,
    ANIMAL_COSTS,
    CROPS,
    PRODUCTS,
    SEED_COSTS,
    CandidateGenerator,
)

from .joint_router_dataset import DECISION_IDS, extract_decision_features


UNIT_OPERATIONS = frozenset(
    {
        "PASS",
        "NORTH",
        "SOUTH",
        "EAST",
        "WEST",
        "PLANT",
        "WATER",
        "HARVEST",
        "FERTILIZE",
        "BUILD_COOP",
        "BUILD_PASTURE",
        "DIG",
        "FEED",
        "CARE",
        "COLLECT_FERTILIZER",
        "PICKUP",
        "DROP",
        "PLACE",
    }
)
MARKET_OPERATIONS = frozenset(
    {"NO_ORDER", "BUY_SEED", "BUY_ANIMAL", "BUY_PRODUCT", "SELL", "HIRE", "BUY_LAND"}
)


class JointEnsemblePolicy:
    """Turn per-slot probabilities into one resource-consistent joint action.

    Candidate masking proves unit actions from the acting-seat observation.  The
    market queue reserves money and shed stock after every selection.  Exact
    orders cannot repeat, and decoding stops at one model-selected terminator.
    """

    def __init__(self, bundle: Any, *, max_orders: int = 10) -> None:
        self.bundle = bundle
        self.max_orders = int(max_orders)
        self.generator = CandidateGenerator()
        self.decode_stats: Counter[str] = Counter()

    def _probabilities(self, features: np.ndarray) -> dict[int, float]:
        probabilities = np.asarray(self.bundle.predict_proba(features[None, :])[0])
        return {
            int(label): float(probability)
            for label, probability in zip(np.asarray(self.bundle.classes_), probabilities)
        }

    def _probability_rows(self, features: list[np.ndarray]) -> list[dict[int, float]]:
        matrix = np.stack(features)
        probabilities = np.asarray(self.bundle.predict_proba(matrix))
        classes = np.asarray(self.bundle.classes_)
        return [
            {int(label): float(probability) for label, probability in zip(classes, row)}
            for row in probabilities
        ]

    @staticmethod
    def _candidate_utility(action: tuple[Any, ...], observation: Mapping[str, Any]) -> tuple[float, str]:
        operation = str(action[0])
        if operation == "PLANT" and len(action) > 1:
            prices = (observation.get("market", {}) or {}).get("prices", {}) or {}
            crop = str(action[1])
            return (float(prices.get(crop, 0) or 0) / max(1, SEED_COSTS[crop]), crop)
        if operation == "PICKUP" and len(action) > 1:
            priorities = {"GOOSE": 7, "COW": 6, "SHEEP": 5, "FERTILIZER": 4, "WHEAT": 3}
            return (float(priorities.get(str(action[1]), 1)), str(action[1]))
        return (0.0, repr(action))

    def _unit_action(
        self,
        observation: Mapping[str, Any],
        unit_index: int,
        reserved_seeds: Counter[str],
        reserved_shed: Counter[str],
        probabilities: dict[int, float] | None = None,
    ) -> list[Any]:
        slot = "farmer" if unit_index == 0 else "hand"
        if probabilities is None:
            features = extract_decision_features(
                observation, slot=slot, unit_index=unit_index
            )
            probabilities = self._probabilities(features)
        candidates = list(self.generator.unit_candidates(observation, unit_index).candidates)
        private = observation.get("private", {}) or {}
        seeds = private.get("seeds", {}) or {}
        shed = private.get("shed", {}) or {}
        feasible = []
        for candidate in candidates:
            operation = str(candidate.action[0])
            if operation not in UNIT_OPERATIONS:
                continue
            if operation == "PLANT":
                item = str(candidate.action[1])
                if reserved_seeds[item] >= int(seeds.get(item, 0) or 0):
                    continue
            if operation == "PICKUP":
                item = str(candidate.action[1])
                amount = int(candidate.action[2]) if len(candidate.action) > 2 else 1
                if reserved_shed[item] + amount > int(shed.get(item, 0) or 0):
                    continue
            score = probabilities.get(DECISION_IDS[operation], 0.0)
            feasible.append((score, self._candidate_utility(candidate.action, observation), candidate.action))
        if not feasible:
            return ["PASS"]
        _, _, chosen = max(feasible, key=lambda value: (value[0], value[1]))
        if chosen[0] == "PLANT":
            reserved_seeds[str(chosen[1])] += 1
        elif chosen[0] == "PICKUP":
            amount = int(chosen[2]) if len(chosen) > 2 else 1
            reserved_shed[str(chosen[1])] += amount
        self.decode_stats[str(chosen[0])] += 1
        return list(chosen)

    @staticmethod
    def _fib(index: int) -> int:
        a, b = 1, 1
        for _ in range(max(0, index)):
            a, b = b, a + b
        return a

    def _market_candidates(
        self,
        observation: Mapping[str, Any],
        *,
        budget: int,
        shed_remaining: Counter[str],
        hires_today: int,
        unlocked: int,
        seen: set[tuple[Any, ...]],
        seen_operations: set[str],
    ) -> list[tuple[list[Any], int]]:
        seat = int(observation["player"])
        farm = observation["farms"][seat]
        private = observation.get("private", {}) or {}
        seeds = private.get("seeds", {}) or {}
        market = observation.get("market", {}) or {}
        prices = market.get("prices", {}) or {}
        result: list[tuple[list[Any], int]] = [(["NO_ORDER"], 0)]
        inventories = private.get("inventories", []) or []
        owned = Counter({str(key): int(value or 0) for key, value in (private.get("shed", {}) or {}).items()})
        for inventory in inventories:
            if isinstance(inventory, Mapping):
                owned.update({str(key): int(value or 0) for key, value in inventory.items()})
        board_animals = {
            str(tile.get("animal"))
            for row in (farm.get("tiles", []) or [])
            for tile in row
            if isinstance(tile, Mapping) and tile.get("animal")
        }

        for item in sorted(PRODUCTS, key=lambda name: (-int(prices.get(name, 0) or 0), name)):
            quantity = min(20, int(shed_remaining.get(item, 0) or 0))
            if quantity > 0 and "SELL" not in seen_operations:
                result.append((["SELL", item, quantity], 0))
        for crop in sorted(CROPS, key=lambda name: (int(seeds.get(name, 0) or 0), name)):
            cost = SEED_COSTS[crop]
            if cost <= budget and int(seeds.get(crop, 0) or 0) < 3 and "BUY_SEED" not in seen_operations:
                result.append((["BUY_SEED", crop, 1], cost))
        for animal in ANIMALS:
            cost = ANIMAL_COSTS[animal]
            if (
                cost <= budget
                and owned[animal] == 0
                and animal not in board_animals
                and "BUY_ANIMAL" not in seen_operations
            ):
                result.append((["BUY_ANIMAL", animal, 1], cost))
        for item in ("WHEAT", "FERTILIZER"):
            # The official engine quotes post-buy inventory.  A single unit and
            # a 10% reserve conservatively absorb one-step price movement.
            cost = int(np.ceil(1.1 * float(prices.get(item, 0) or 0)))
            target = 4 if item == "WHEAT" else 1
            if cost <= budget and owned[item] < target and "BUY_PRODUCT" not in seen_operations:
                result.append((["BUY_PRODUCT", item, 1], cost))
        hire_cost = self._fib(hires_today)
        if (
            hire_cost <= budget
            and int(observation.get("hour", 0)) == 0
            and len(farm.get("hands", []) or []) < 2
            and "HIRE" not in seen_operations
        ):
            result.append((["HIRE"], hire_cost))
        if unlocked < 4 and "BUY_LAND" not in seen_operations:
            land_cost = (1000, 2000, 4000)[min(unlocked - 1, 2)]
            if land_cost <= budget:
                result.append((["BUY_LAND"], land_cost))
        return [(action, cost) for action, cost in result if tuple(action) not in seen]

    def _market(self, observation: Mapping[str, Any]) -> list[list[Any]]:
        seat = int(observation["player"])
        farm = observation["farms"][seat]
        private = observation.get("private", {}) or {}
        budget = int(farm.get("money", 0) or 0)
        shed_remaining: Counter[str] = Counter(
            {str(key): int(value or 0) for key, value in (private.get("shed", {}) or {}).items()}
        )
        hires_today = int(farm.get("hires_today", 0) or 0)
        unlocked = len(farm.get("unlocked_quadrants", []) or ["NW"])
        seen: set[tuple[Any, ...]] = set()
        seen_operations: set[str] = set()
        result: list[list[Any]] = []
        probability_rows = self._probability_rows(
            [
                extract_decision_features(
                    observation, slot="market", market_order_index=order_index
                )
                for order_index in range(self.max_orders)
            ]
        )
        for order_index in range(self.max_orders):
            probabilities = probability_rows[order_index]
            candidates = self._market_candidates(
                observation,
                budget=budget,
                shed_remaining=shed_remaining,
                hires_today=hires_today,
                unlocked=unlocked,
                seen=seen,
                seen_operations=seen_operations,
            )
            if not candidates:
                break
            scored = [
                (probabilities.get(DECISION_IDS[str(action[0])], 0.0), -cost, repr(action), action, cost)
                for action, cost in candidates
            ]
            _, _, _, chosen, cost = max(scored)
            if chosen[0] == "NO_ORDER":
                break
            key = tuple(chosen)
            seen.add(key)
            seen_operations.add(str(chosen[0]))
            result.append(chosen)
            budget -= cost
            if chosen[0] == "SELL":
                shed_remaining[str(chosen[1])] -= int(chosen[2])
            elif chosen[0] == "HIRE":
                hires_today += 1
            elif chosen[0] == "BUY_LAND":
                unlocked += 1
            self.decode_stats[str(chosen[0])] += 1
        return result

    def __call__(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        seat = int(observation["player"])
        hand_count = len(observation["farms"][seat].get("hands", []) or [])
        reserved_seeds: Counter[str] = Counter()
        reserved_shed: Counter[str] = Counter()
        unit_probability_rows = self._probability_rows(
            [
                extract_decision_features(
                    observation,
                    slot="farmer" if index == 0 else "hand",
                    unit_index=index,
                )
                for index in range(hand_count + 1)
            ]
        )
        actions = [
            self._unit_action(
                observation,
                index,
                reserved_seeds,
                reserved_shed,
                unit_probability_rows[index],
            )
            for index in range(hand_count + 1)
        ]
        market = self._market(observation)
        if len(market) != len({tuple(order) for order in market}):
            raise RuntimeError("decoder produced duplicate market orders")
        return {"farmer": actions[0], "hands": actions[1:], "market": market}
