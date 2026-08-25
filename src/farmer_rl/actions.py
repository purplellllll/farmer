"""Conservative legal candidates, masks, and a fixed-shape joint-action codec.

The official engine silently converts illegal operations to no-ops.  Generating
state-conditioned candidates before scoring therefore improves both exploration
efficiency and data quality.  This module is conservative rather than complete:
omitting a legal action is safer than labelling an illegal one as legal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .environment import Action
from .errors import InvalidActionError, SeatSafetyError

CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")
PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
MOVES = ("NORTH", "SOUTH", "EAST", "WEST")
SEED_COSTS = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
ANIMAL_COSTS = {"GOOSE": 300, "COW": 400, "SHEEP": 500}


@dataclass(frozen=True)
class ActionCandidate:
    slot: str
    action: tuple[Any, ...]
    rationale: str

    def as_list(self) -> list[Any]:
        return list(self.action)


@dataclass(frozen=True)
class CandidateSet:
    slot: str
    candidates: tuple[ActionCandidate, ...]
    capacity: int

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("every slot must contain at least one candidate")
        if len(self.candidates) > self.capacity:
            raise ValueError(f"{self.slot} has {len(self.candidates)} candidates, capacity={self.capacity}")

    @property
    def mask(self) -> tuple[int, ...]:
        return (1,) * len(self.candidates) + (0,) * (self.capacity - len(self.candidates))

    def choose(self, index: int) -> list[Any]:
        if index < 0 or index >= len(self.candidates):
            raise InvalidActionError(f"masked or out-of-range action index {index} for {self.slot}")
        return self.candidates[index].as_list()


class CandidateGenerator:
    """Generate per-unit and market candidates from a player's legal observation."""

    def __init__(self, *, capacity: int = 64, market_quantity_cap: int = 20) -> None:
        if capacity < 8:
            raise ValueError("capacity must leave room for common primitive actions")
        self.capacity = int(capacity)
        self.market_quantity_cap = int(market_quantity_cap)

    @staticmethod
    def _own_parts(observation: Mapping[str, Any]) -> tuple[int, Mapping[str, Any], Mapping[str, Any]]:
        seat = int(observation.get("player", -1))
        farms = observation.get("farms", [])
        if seat not in (0, 1) or not isinstance(farms, Sequence) or len(farms) != 2:
            raise SeatSafetyError("candidate generation requires a two-farm observation with player=0 or 1")
        farm = farms[seat]
        private = observation.get("private", {}) or {}
        if not isinstance(farm, Mapping) or not isinstance(private, Mapping):
            raise ValueError("farm/private state must be mappings")
        return seat, farm, private

    @staticmethod
    def _positive_items(inventory: Mapping[str, Any]) -> list[str]:
        return sorted(str(key) for key, value in inventory.items() if int(value or 0) > 0)

    @staticmethod
    def _dedupe(items: Iterable[ActionCandidate]) -> tuple[ActionCandidate, ...]:
        seen: set[tuple[Any, ...]] = set()
        result: list[ActionCandidate] = []
        for item in items:
            if item.action not in seen:
                seen.add(item.action)
                result.append(item)
        return tuple(result)

    def unit_candidates(self, observation: Mapping[str, Any], unit_index: int) -> CandidateSet:
        """Return candidates for farmer index 0 or hand index ``unit_index - 1``."""

        _, farm, private = self._own_parts(observation)
        positions = [farm.get("farmer", [0, 0]), *(farm.get("hands", []) or [])]
        inventories = private.get("inventories", []) or []
        if unit_index < 0 or unit_index >= len(positions):
            raise IndexError(f"unit_index {unit_index} is not active")
        position = positions[unit_index]
        x, y = int(position[0]), int(position[1])
        board = farm.get("tiles", []) or []
        height = len(board)
        width = len(board[0]) if height else 0
        tile = board[y][x] if 0 <= y < height and 0 <= x < width else None
        carried = inventories[unit_index] if unit_index < len(inventories) else {}
        carried = carried if isinstance(carried, Mapping) else {}
        shed = private.get("shed", {}) or {}
        seeds = private.get("seeds", {}) or {}
        slot = "farmer" if unit_index == 0 else f"hand:{unit_index - 1}"
        result = [ActionCandidate(slot, ("PASS",), "always legal")]

        for op, allowed in (
            ("NORTH", y > 0),
            ("SOUTH", y + 1 < height),
            ("EAST", x + 1 < width),
            ("WEST", x > 0),
        ):
            if allowed:
                result.append(ActionCandidate(slot, (op,), "destination is inside the board"))

        is_unlocked = tile != "LOCKED"
        if tile is None and is_unlocked:
            for crop in CROPS:
                if int(seeds.get(crop, 0) or 0) > 0:
                    result.append(ActionCandidate(slot, ("PLANT", crop), "seed available on empty unlocked tile"))
            result.extend(
                (
                    ActionCandidate(slot, ("BUILD_COOP",), "empty unlocked tile"),
                    ActionCandidate(slot, ("BUILD_PASTURE",), "empty unlocked tile"),
                )
            )
        elif isinstance(tile, Mapping):
            kind = str(tile.get("kind", ""))
            if kind == "PLANT":
                if not bool(tile.get("watered_today", False)):
                    result.append(ActionCandidate(slot, ("WATER",), "plant has not been watered today"))
                if int(carried.get("FERTILIZER", 0) or 0) > 0:
                    result.append(ActionCandidate(slot, ("FERTILIZE",), "fertilizer is carried on a plant"))
                if int(tile.get("yield_units", 0) or 0) > 0:
                    result.append(ActionCandidate(slot, ("HARVEST",), "plant has harvestable yield"))
                result.append(ActionCandidate(slot, ("DIG",), "plant can be removed"))
            elif kind == "WEED":
                result.append(ActionCandidate(slot, ("DIG",), "weed can be removed"))
            elif kind in {"COOP", "PASTURE"}:
                animal = tile.get("animal")
                if animal:
                    if not bool(tile.get("fed_today", False)) and int(carried.get("WHEAT", 0) or 0) > 0:
                        result.append(ActionCandidate(slot, ("FEED",), "animal is unfed and wheat is carried"))
                    if not bool(tile.get("cared_today", False)):
                        result.append(ActionCandidate(slot, ("CARE",), "animal has not been cared for today"))
                    if int(tile.get("yield_units", 0) or 0) > 0:
                        result.append(ActionCandidate(slot, ("HARVEST",), "animal structure has yield"))
                    if int(tile.get("fertilizer_available", 0) or 0) > 0:
                        result.append(ActionCandidate(slot, ("COLLECT_FERTILIZER",), "fertilizer is available"))
                else:
                    compatible = ("GOOSE",) if kind == "COOP" else ("COW", "SHEEP")
                    for animal_name in compatible:
                        if int(carried.get(animal_name, 0) or 0) > 0:
                            result.append(ActionCandidate(slot, ("PLACE", animal_name), "compatible animal is carried"))
                    result.append(ActionCandidate(slot, ("DIG",), "empty structure can be removed"))

        half = width // 2
        shed_adjacent = {(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)}
        if (x, y) in shed_adjacent:
            if any(int(value or 0) > 0 for value in carried.values()):
                result.append(ActionCandidate(slot, ("DROP",), "unit is shed-adjacent and carries inventory"))
            for item in self._positive_items(shed):
                result.append(ActionCandidate(slot, ("PICKUP", item, 1), "shed contains item"))

        candidates = self._dedupe(result)
        return CandidateSet(slot, candidates[: self.capacity], self.capacity)

    @staticmethod
    def _fib(index: int) -> int:
        a, b = 1, 1
        for _ in range(max(0, index)):
            a, b = b, a + b
        return a

    def market_candidates(self, observation: Mapping[str, Any], order_index: int) -> CandidateSet:
        _, farm, private = self._own_parts(observation)
        if not 0 <= order_index < 10:
            raise IndexError("market order index must be in [0, 9]")
        slot = f"market:{order_index}"
        result = [ActionCandidate(slot, ("NO_ORDER",), "leave this queue slot empty")]
        shed = private.get("shed", {}) or {}
        money = int(farm.get("money", 0) or 0)

        for item in PRODUCTS:
            quantity = min(int(shed.get(item, 0) or 0), self.market_quantity_cap)
            if quantity > 0:
                result.append(ActionCandidate(slot, ("SELL", item, quantity), "item is currently in the shed"))
        # Fixed-cost purchase candidates can be proven affordable from observation.
        for crop in CROPS:
            if money >= SEED_COSTS[crop]:
                result.append(ActionCandidate(slot, ("BUY_SEED", crop, 1), "cash covers the fixed seed price"))
        for animal in ANIMALS:
            if money >= ANIMAL_COSTS[animal]:
                result.append(ActionCandidate(slot, ("BUY_ANIMAL", animal, 1), "cash covers the fixed animal price"))
        # BUY_PRODUCT quotes the post-buy dynamic price, which is not present as a
        # guaranteed upper bound in observation. It stays outside the conservative
        # mask until an exact market-state simulator is plugged into the decoder.
        hires_today = int(farm.get("hires_today", 0) or 0)
        if money >= self._fib(hires_today):
            result.append(ActionCandidate(slot, ("HIRE",), "current cash covers the next Fibonacci hire"))
        quadrants = set(farm.get("unlocked_quadrants", []) or ["NW"])
        next_land_index = max(0, len(quadrants) - 1)
        land_costs = (1000, 2000, 4000)
        if len(quadrants) < 4 and money >= land_costs[min(next_land_index, 2)]:
            result.append(ActionCandidate(slot, ("BUY_LAND",), "locked quadrant remains and current cash covers base cost"))
        candidates = self._dedupe(result)
        return CandidateSet(slot, candidates[: self.capacity], self.capacity)

    def joint_candidates(self, observation: Mapping[str, Any], *, max_hands: int, max_orders: int = 10) -> tuple[CandidateSet, ...]:
        _, farm, _ = self._own_parts(observation)
        active_hands = len(farm.get("hands", []) or [])
        sets: list[CandidateSet] = [self.unit_candidates(observation, 0)]
        for hand_index in range(max_hands):
            if hand_index < active_hands:
                sets.append(self.unit_candidates(observation, hand_index + 1))
            else:
                slot = f"hand:{hand_index}"
                sets.append(CandidateSet(slot, (ActionCandidate(slot, ("PASS",), "inactive padded hand"),), self.capacity))
        for order_index in range(max_orders):
            sets.append(self.market_candidates(observation, order_index))
        return tuple(sets)


class JointActionCodec:
    """Map fixed-shape slot indices to the official variable-length action dict."""

    def __init__(self, generator: CandidateGenerator, *, max_hands: int = 16, max_orders: int = 10) -> None:
        self.generator = generator
        self.max_hands = int(max_hands)
        self.max_orders = int(max_orders)

    @property
    def slots(self) -> int:
        return 1 + self.max_hands + self.max_orders

    def candidates(self, observation: Mapping[str, Any]) -> tuple[CandidateSet, ...]:
        return self.generator.joint_candidates(
            observation, max_hands=self.max_hands, max_orders=self.max_orders
        )

    def mask(self, observation: Mapping[str, Any]) -> tuple[int, ...]:
        return tuple(value for candidate_set in self.candidates(observation) for value in candidate_set.mask)

    def decode(self, observation: Mapping[str, Any], indices: Sequence[int]) -> Action:
        sets = self.candidates(observation)
        if len(indices) != len(sets):
            raise InvalidActionError(f"expected {len(sets)} slot indices, got {len(indices)}")
        chosen = [candidate_set.choose(int(index)) for candidate_set, index in zip(sets, indices)]
        _, farm, _ = self.generator._own_parts(observation)
        active_hands = len(farm.get("hands", []) or [])
        proposed_market = [
            action
            for action in chosen[1 + self.max_hands :]
            if action and action[0] != "NO_ORDER"
        ][: self.max_orders]
        # Reserve currently-observable resources across the whole ordered queue.
        # We do not credit sale proceeds because exact interleaving with the
        # opponent changes realized prices; this deliberately errs on safety.
        _, farm, private = self.generator._own_parts(observation)
        available = {str(key): int(value or 0) for key, value in (private.get("shed", {}) or {}).items()}
        budget = int(farm.get("money", 0) or 0)
        hires_today = int(farm.get("hires_today", 0) or 0)
        unlocked = len(farm.get("unlocked_quadrants", []) or ["NW"])
        market: list[list[Any]] = []
        for action in proposed_market:
            op = action[0]
            if op == "SELL":
                item, requested = str(action[1]), int(action[2])
                quantity = min(requested, available.get(item, 0))
                if quantity <= 0:
                    continue
                available[item] = available.get(item, 0) - quantity
                market.append(["SELL", item, quantity])
            elif op == "BUY_SEED":
                cost = SEED_COSTS[str(action[1])] * int(action[2])
                if cost <= budget:
                    budget -= cost
                    market.append(action)
            elif op == "BUY_ANIMAL":
                cost = ANIMAL_COSTS[str(action[1])] * int(action[2])
                if cost <= budget:
                    budget -= cost
                    market.append(action)
            elif op == "HIRE":
                cost = self.generator._fib(hires_today)
                if cost <= budget:
                    budget -= cost
                    hires_today += 1
                    market.append(action)
            elif op == "BUY_LAND" and unlocked < 4:
                cost = (1000, 2000, 4000)[min(unlocked - 1, 2)]
                if cost <= budget:
                    budget -= cost
                    unlocked += 1
                    market.append(action)
        return {
            "farmer": chosen[0],
            "hands": chosen[1 : 1 + active_hands],
            "market": market,
        }

    def encode(self, observation: Mapping[str, Any], action: Mapping[str, Any]) -> tuple[int, ...]:
        """Encode an expert action; raises when it is outside the conservative set."""

        sets = self.candidates(observation)
        _, farm, _ = self.generator._own_parts(observation)
        active_hands = len(farm.get("hands", []) or [])
        targets: list[Sequence[Any]] = [action.get("farmer", ["PASS"])]
        provided_hands = list(action.get("hands", []) or [])
        targets.extend(provided_hands[:active_hands])
        targets.extend([["PASS"]] * (self.max_hands - active_hands))
        provided_market = list(action.get("market", []) or [])[: self.max_orders]
        targets.extend(provided_market)
        targets.extend([["NO_ORDER"]] * (self.max_orders - len(provided_market)))
        indices: list[int] = []
        for candidate_set, target in zip(sets, targets):
            target_tuple = tuple(target)
            for index, candidate in enumerate(candidate_set.candidates):
                if candidate.action == target_tuple:
                    indices.append(index)
                    break
            else:
                raise InvalidActionError(f"expert action {target_tuple!r} absent from {candidate_set.slot}")
        return tuple(indices)
