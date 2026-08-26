"""Conservative legal candidates, masks, and a fixed-shape joint-action codec.

The official engine silently converts illegal operations to no-ops.  Generating
state-conditioned candidates before scoring therefore improves both exploration
efficiency and data quality.  This module is conservative rather than complete:
omitting a legal action is safer than labelling an illegal one as legal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

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
                quantity = min(int(shed.get(item, 0) or 0), 6)
                if quantity > 1:
                    result.append(
                        ActionCandidate(
                            slot,
                            ("PICKUP", item, quantity),
                            "bounded curriculum pickup does not exceed current shed stock",
                        )
                    )

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
        # Keep the original candidates above in their historical order so old
        # BC checkpoints remain a useful warm start.  Curriculum-v2 uses a
        # deliberately wider but still bounded set of purchase quantities.
        # These candidates are appended rather than inserted for that reason.
        for crop in CROPS:
            quantity = 3
            if money >= SEED_COSTS[crop] * quantity:
                result.append(
                    ActionCandidate(
                        slot,
                        ("BUY_SEED", crop, quantity),
                        "curriculum quantity is affordable at its fixed unit cost",
                    )
                )
        market = observation.get("market", {}) or {}
        market_inventory = market.get("inventory", {}) or {}
        market_prices = market.get("prices", {}) or {}
        shed_used = sum(max(0, int(value or 0)) for value in shed.values())
        # The official engine permits BUY_PRODUCT only for WHEAT and
        # FERTILIZER.  Quantity four is the curriculum-v2 expert action.  A
        # public-price reserve is rechecked by the sequential compiler below.
        for item in ("WHEAT", "FERTILIZER"):
            for quantity in (1, 4):
                price = max(1, int(market_prices.get(item, 0) or 0))
                supply = max(0, int(market_inventory.get(item, 0) or 0))
                if shed_used + quantity <= 100 and supply >= quantity and money >= price * quantity:
                    result.append(
                        ActionCandidate(
                            slot,
                            ("BUY_PRODUCT", item, quantity),
                            "curriculum quantity fits public supply, shed and current-price budget",
                        )
                    )
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

    def select(
        self,
        observation: Mapping[str, Any],
        chooser: Callable[[int, CandidateSet, tuple[int, ...]], int],
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Choose a canonical joint action with prefix-conditioned masks.

        The policy still scores all fixed output slots in one Transformer
        forward pass, but later slots may only select actions that remain
        executable after earlier selections reserve shared seeds, shed stock,
        cash, market inventory and queue semantics.  Returning the exact masks
        used by the chooser lets PPO recompute the same conditional likelihood
        during its update instead of assigning credit to actions discarded by
        :meth:`decode`.
        """

        sets = self.candidates(observation)
        _, farm, private = self.generator._own_parts(observation)
        active_hands = len(farm.get("hands", []) or [])
        unit_slot_count = 1 + self.max_hands
        available = {
            str(key): max(0, int(value or 0))
            for key, value in (private.get("shed", {}) or {}).items()
        }
        seeds = {
            str(key): max(0, int(value or 0))
            for key, value in (private.get("seeds", {}) or {}).items()
        }
        selected: list[int] = []
        masks: list[tuple[int, ...]] = []

        def choose(slot_index: int, candidate_set: CandidateSet, row: list[int]) -> int:
            row_tuple = tuple(int(value) for value in row)
            index = int(chooser(slot_index, candidate_set, row_tuple))
            if index < 0 or index >= candidate_set.capacity or not row_tuple[index]:
                raise InvalidActionError(
                    f"chooser selected masked index {index} for {candidate_set.slot}"
                )
            selected.append(index)
            masks.append(row_tuple)
            return index

        unit_actions: list[tuple[Any, ...]] = []
        for slot_index, candidate_set in enumerate(sets[:unit_slot_count]):
            row = list(candidate_set.mask)
            if slot_index <= active_hands:
                for candidate_index, candidate in enumerate(candidate_set.candidates):
                    action = candidate.action
                    if action[0] == "PLANT" and seeds.get(str(action[1]), 0) <= 0:
                        row[candidate_index] = 0
                    elif action[0] == "PICKUP":
                        item = str(action[1])
                        requested = int(action[2]) if len(action) >= 3 else 1
                        if requested <= 0 or requested > available.get(item, 0):
                            row[candidate_index] = 0
            index = choose(slot_index, candidate_set, row)
            action = candidate_set.candidates[index].action
            unit_actions.append(action)
            if action[0] == "PLANT":
                crop = str(action[1])
                seeds[crop] = seeds.get(crop, 0) - 1
            elif action[0] == "PICKUP":
                item = str(action[1])
                requested = int(action[2]) if len(action) >= 3 else 1
                available[item] = available.get(item, 0) - requested

        budget = int(farm.get("money", 0) or 0)
        hires_today = int(farm.get("hires_today", 0) or 0)
        unlocked = len(farm.get("unlocked_quadrants", []) or ["NW"])
        public_market = observation.get("market", {}) or {}
        market_inventory = {
            str(key): max(0, int(value or 0))
            for key, value in (public_market.get("inventory", {}) or {}).items()
        }
        market_prices = {
            str(key): max(1, int(value or 0))
            for key, value in (public_market.get("prices", {}) or {}).items()
        }
        inventories = private.get("inventories", []) or []
        pending_drop = 0
        for unit_index, action in enumerate(unit_actions[: 1 + active_hands]):
            if action[0] == "DROP" and unit_index < len(inventories):
                carried = inventories[unit_index]
                if isinstance(carried, Mapping):
                    pending_drop += sum(max(0, int(value or 0)) for value in carried.values())
        shed_room = max(0, 100 - sum(available.values()) - pending_drop)
        seen_orders: set[tuple[Any, ...]] = set()
        market_open = True

        for slot_index, candidate_set in enumerate(sets[unit_slot_count:], start=unit_slot_count):
            row = [0] * candidate_set.capacity
            row[0] = 1
            if market_open:
                for candidate_index, candidate in enumerate(candidate_set.candidates[1:], start=1):
                    action = candidate.action
                    op = str(action[0])
                    semantic_key = (op, str(action[1])) if len(action) >= 2 else (op,)
                    valid = semantic_key not in seen_orders
                    if op == "SELL":
                        valid = valid and int(action[2]) <= available.get(str(action[1]), 0)
                    elif op == "BUY_SEED":
                        valid = valid and SEED_COSTS[str(action[1])] * int(action[2]) <= budget
                    elif op == "BUY_ANIMAL":
                        quantity = int(action[2])
                        valid = (
                            valid
                            and ANIMAL_COSTS[str(action[1])] * quantity <= budget
                            and quantity <= shed_room
                        )
                    elif op == "BUY_PRODUCT":
                        item, quantity = str(action[1]), int(action[2])
                        valid = (
                            valid
                            and item in {"WHEAT", "FERTILIZER"}
                            and quantity <= shed_room
                            and quantity <= market_inventory.get(item, 0)
                            and market_prices.get(item, 1) * quantity <= budget
                        )
                    elif op == "HIRE":
                        valid = valid and self.generator._fib(hires_today) <= budget
                    elif op == "BUY_LAND":
                        cost = (1000, 2000, 4000)[min(max(0, unlocked - 1), 2)]
                        valid = valid and unlocked < 4 and cost <= budget
                    else:
                        valid = False
                    row[candidate_index] = int(valid)

            index = choose(slot_index, candidate_set, row)
            action = candidate_set.candidates[index].action
            if index == 0:
                market_open = False
                continue
            op = str(action[0])
            semantic_key = (op, str(action[1])) if len(action) >= 2 else (op,)
            seen_orders.add(semantic_key)
            if op == "SELL":
                item, quantity = str(action[1]), int(action[2])
                available[item] = available.get(item, 0) - quantity
            elif op == "BUY_SEED":
                budget -= SEED_COSTS[str(action[1])] * int(action[2])
            elif op == "BUY_ANIMAL":
                quantity = int(action[2])
                budget -= ANIMAL_COSTS[str(action[1])] * quantity
                shed_room -= quantity
            elif op == "BUY_PRODUCT":
                item, quantity = str(action[1]), int(action[2])
                budget -= market_prices.get(item, 1) * quantity
                shed_room -= quantity
                market_inventory[item] = market_inventory.get(item, 0) - quantity
            elif op == "HIRE":
                budget -= self.generator._fib(hires_today)
                hires_today += 1
            elif op == "BUY_LAND":
                budget -= (1000, 2000, 4000)[min(max(0, unlocked - 1), 2)]
                unlocked += 1

        return tuple(selected), tuple(value for row in masks for value in row)

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
        unit_actions = [list(action) for action in chosen[: 1 + active_hands]]
        for index, action in enumerate(unit_actions):
            if action and action[0] == "PICKUP" and len(action) >= 3:
                item, requested = str(action[1]), int(action[2])
                quantity = min(requested, available.get(item, 0))
                if quantity <= 0:
                    unit_actions[index] = ["PASS"]
                    continue
                unit_actions[index] = ["PICKUP", item, quantity]
                available[item] = available.get(item, 0) - quantity
        budget = int(farm.get("money", 0) or 0)
        hires_today = int(farm.get("hires_today", 0) or 0)
        unlocked = len(farm.get("unlocked_quadrants", []) or ["NW"])
        public_market = observation.get("market", {}) or {}
        market_inventory = {
            str(key): int(value or 0)
            for key, value in (public_market.get("inventory", {}) or {}).items()
        }
        market_prices = {
            str(key): max(1, int(value or 0))
            for key, value in (public_market.get("prices", {}) or {}).items()
        }
        inventories = private.get("inventories", []) or []
        pending_drop = 0
        for unit_index, action in enumerate(unit_actions):
            if action and action[0] == "DROP" and unit_index < len(inventories):
                carried = inventories[unit_index]
                if isinstance(carried, Mapping):
                    pending_drop += sum(max(0, int(value or 0)) for value in carried.values())
        shed_room = max(
            0,
            100 - sum(max(0, value) for value in available.values()) - pending_drop,
        )
        # A joint action is compiled in order.  Candidate slots are scored
        # independently, but duplicate semantic orders are never emitted and
        # every accepted order reserves its shared resources for later slots.
        # This prevents the old ten-identical-BUY_SEED collapse even when all
        # market heads choose the same candidate index.
        seen_orders: set[tuple[Any, ...]] = set()
        market: list[list[Any]] = []
        for action in proposed_market:
            op = action[0]
            semantic_key = (op, str(action[1])) if len(action) >= 2 else (op,)
            if semantic_key in seen_orders:
                continue
            if op == "SELL":
                item, requested = str(action[1]), int(action[2])
                quantity = min(requested, available.get(item, 0))
                if quantity <= 0:
                    continue
                available[item] = available.get(item, 0) - quantity
                market.append(["SELL", item, quantity])
                seen_orders.add(semantic_key)
            elif op == "BUY_SEED":
                cost = SEED_COSTS[str(action[1])] * int(action[2])
                if cost <= budget:
                    budget -= cost
                    market.append(action)
                    seen_orders.add(semantic_key)
            elif op == "BUY_ANIMAL":
                cost = ANIMAL_COSTS[str(action[1])] * int(action[2])
                quantity = int(action[2])
                if cost <= budget and quantity <= shed_room:
                    budget -= cost
                    shed_room -= quantity
                    market.append(action)
                    seen_orders.add(semantic_key)
            elif op == "BUY_PRODUCT":
                item, quantity = str(action[1]), int(action[2])
                # Reserve the visible quote for every unit.  The official
                # engine requotes after each unit and still performs its own
                # final affordability check; this compiler never relies on
                # same-turn sale proceeds.
                cost = market_prices.get(item, 1) * quantity
                if (
                    item in {"WHEAT", "FERTILIZER"}
                    and quantity <= shed_room
                    and quantity <= market_inventory.get(item, 0)
                    and cost <= budget
                ):
                    budget -= cost
                    shed_room -= quantity
                    market_inventory[item] -= quantity
                    market.append(action)
                    seen_orders.add(semantic_key)
            elif op == "HIRE":
                cost = self.generator._fib(hires_today)
                if cost <= budget:
                    budget -= cost
                    hires_today += 1
                    market.append(action)
                    seen_orders.add(semantic_key)
            elif op == "BUY_LAND" and unlocked < 4:
                cost = (1000, 2000, 4000)[min(unlocked - 1, 2)]
                if cost <= budget:
                    budget -= cost
                    unlocked += 1
                    market.append(action)
                    seen_orders.add(semantic_key)
        return {
            "farmer": unit_actions[0],
            "hands": unit_actions[1:],
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
