"""Deterministic public-observation tokenization for residual Transformers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .actions import ANIMALS, CROPS, PRODUCTS
from .errors import SeatSafetyError

FEATURE_DIM = 32
TOKEN_TYPES = {
    "PAD": 0,
    "GLOBAL": 1,
    "TILE": 2,
    "UNIT": 3,
    "PRIVATE_RESOURCE": 4,
    "MARKET_RESOURCE": 5,
    "SHOP": 6,
}
TILE_KINDS = {"EMPTY": 0, "LOCKED": 1, "PLANT": 2, "WEED": 3, "COOP": 4, "PASTURE": 5}
RESOURCE_VOCAB = tuple(dict.fromkeys((*PRODUCTS, *CROPS, *ANIMALS)))
RESOURCE_IDS = {name: index + 1 for index, name in enumerate(RESOURCE_VOCAB)}
SHOP_VOCAB = ("BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE", "SMOOTHIE_SHOP")
SHOP_IDS = {name: index + 1 for index, name in enumerate(SHOP_VOCAB)}


@dataclass(frozen=True)
class TokenBatch:
    values: tuple[tuple[float, ...], ...]
    token_type_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    acting_seat: int

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.values), FEATURE_DIM

    def flat_values(self) -> tuple[float, ...]:
        return tuple(value for row in self.values for value in row)


class ObservationTokenizer:
    """Create fixed-length tokens, always ordering the acting player's farm first.

    Feature 0 is token type, feature 1 is relative owner (1=self, -1=opponent),
    and remaining positions are shared normalized numeric/categorical fields.
    Unknown keys remain zero rather than being hashed, keeping train/deploy parity.
    """

    def __init__(self, *, max_tokens: int = 320, max_hands_per_farm: int = 24) -> None:
        if max_tokens < 220:
            raise ValueError("max_tokens must accommodate two default 10x10 farms")
        self.max_tokens = int(max_tokens)
        self.max_hands_per_farm = int(max_hands_per_farm)

    @staticmethod
    def _number(value: Any, scale: float = 1.0) -> float:
        try:
            return float(value or 0.0) / scale
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _row(token_type: str, owner: float = 0.0) -> list[float]:
        result = [0.0] * FEATURE_DIM
        result[0] = float(TOKEN_TYPES[token_type])
        result[1] = owner
        return result

    def tokenize(self, observation: Mapping[str, Any]) -> TokenBatch:
        seat = int(observation.get("player", -1))
        farms = observation.get("farms", [])
        if seat not in (0, 1) or not isinstance(farms, Sequence) or len(farms) != 2:
            raise SeatSafetyError("tokenizer requires player=0/1 and exactly two public farms")
        private = observation.get("private", {}) or {}
        if not isinstance(private, Mapping):
            raise ValueError("private observation must be a mapping")
        rows: list[list[float]] = []
        types: list[int] = []

        global_row = self._row("GLOBAL")
        global_row[2] = self._number(observation.get("step"), 720.0)
        global_row[3] = self._number(observation.get("day"), 30.0)
        global_row[4] = self._number(observation.get("hour"), 24.0)
        global_row[5] = float(seat)
        rows.append(global_row)
        types.append(TOKEN_TYPES["GLOBAL"])

        for role_index, farm_index in enumerate((seat, 1 - seat)):
            farm = farms[farm_index]
            if not isinstance(farm, Mapping):
                raise ValueError("farm entry must be a mapping")
            owner = 1.0 if role_index == 0 else -1.0
            board = farm.get("tiles", []) or []
            height = len(board)
            width = len(board[0]) if height else 0
            for y, line in enumerate(board):
                for x, tile in enumerate(line):
                    row = self._row("TILE", owner)
                    row[2] = x / max(1, width - 1)
                    row[3] = y / max(1, height - 1)
                    if tile is None:
                        kind = "EMPTY"
                    elif tile == "LOCKED":
                        kind = "LOCKED"
                    elif isinstance(tile, Mapping):
                        kind = str(tile.get("kind", "EMPTY"))
                    else:
                        kind = "EMPTY"
                    row[4] = TILE_KINDS.get(kind, 0) / max(TILE_KINDS.values())
                    if isinstance(tile, Mapping):
                        resource = str(tile.get("crop") or tile.get("animal") or "")
                        row[5] = RESOURCE_IDS.get(resource, 0) / max(1, len(RESOURCE_IDS))
                        row[6] = self._number(tile.get("planted_day"), 30.0)
                        row[7] = self._number(tile.get("placed_day"), 30.0)
                        row[8] = self._number(tile.get("yield_units"), 10.0)
                        row[9] = float(bool(tile.get("watered_today", False)))
                        row[10] = self._number(tile.get("consecutive_unwatered"), 2.0)
                        row[11] = self._number(tile.get("fertilized_until_day"), 30.0)
                        row[12] = float(bool(tile.get("fed_today", False)))
                        row[13] = self._number(tile.get("consecutive_unfed"), 2.0)
                        row[14] = float(bool(tile.get("cared_today", False)))
                        row[15] = self._number(tile.get("fertilizer_available"), 4.0)
                        row[16] = self._number(tile.get("pending_care_bonus"), 8.0)
                        row[17] = self._number(tile.get("max_lifespan_step"), 720.0)
                    rows.append(row)
                    types.append(TOKEN_TYPES["TILE"])

            positions = [farm.get("farmer", [0, 0]), *(farm.get("hands", []) or [])]
            for unit_index, position in enumerate(positions[: 1 + self.max_hands_per_farm]):
                row = self._row("UNIT", owner)
                row[2] = self._number(position[0], max(1, width - 1))
                row[3] = self._number(position[1], max(1, height - 1))
                row[4] = 1.0 if unit_index == 0 else 0.0
                row[5] = unit_index / max(1, self.max_hands_per_farm)
                row[6] = self._number(farm.get("money"), 200_000.0)
                row[7] = self._number(farm.get("hires_today"), 24.0)
                row[8] = len(farm.get("unlocked_quadrants", []) or []) / 4.0
                rows.append(row)
                types.append(TOKEN_TYPES["UNIT"])

        shed = private.get("shed", {}) or {}
        seeds = private.get("seeds", {}) or {}
        inventories = private.get("inventories", []) or []
        carried_totals: dict[str, int] = {}
        for inventory in inventories:
            if isinstance(inventory, Mapping):
                for item, count in inventory.items():
                    carried_totals[str(item)] = carried_totals.get(str(item), 0) + int(count or 0)
        for resource in RESOURCE_VOCAB:
            row = self._row("PRIVATE_RESOURCE", 1.0)
            row[2] = RESOURCE_IDS[resource] / max(1, len(RESOURCE_IDS))
            row[3] = self._number(shed.get(resource), 100.0)
            row[4] = self._number(seeds.get(resource), 100.0)
            row[5] = self._number(carried_totals.get(resource), 100.0)
            rows.append(row)
            types.append(TOKEN_TYPES["PRIVATE_RESOURCE"])

        market = observation.get("market", {}) or {}
        market_inventory = market.get("inventory", {}) or {}
        prices = market.get("prices", {}) or {}
        for resource in PRODUCTS:
            row = self._row("MARKET_RESOURCE")
            row[2] = RESOURCE_IDS.get(resource, 0) / max(1, len(RESOURCE_IDS))
            row[3] = self._number(market_inventory.get(resource), 1_000.0)
            row[4] = self._number(prices.get(resource), 500.0)
            rows.append(row)
            types.append(TOKEN_TYPES["MARKET_RESOURCE"])

        for shop in (observation.get("town", {}) or {}).get("unlocked_shops", []) or []:
            row = self._row("SHOP")
            normalized = str(shop).upper().replace(" ", "_")
            row[2] = SHOP_IDS.get(normalized, 0) / max(1, len(SHOP_IDS))
            rows.append(row)
            types.append(TOKEN_TYPES["SHOP"])

        rows = rows[: self.max_tokens]
        types = types[: self.max_tokens]
        real_count = len(rows)
        while len(rows) < self.max_tokens:
            rows.append([0.0] * FEATURE_DIM)
            types.append(TOKEN_TYPES["PAD"])
        return TokenBatch(
            values=tuple(tuple(float(value) for value in row) for row in rows),
            token_type_ids=tuple(types),
            attention_mask=(1,) * real_count + (0,) * (self.max_tokens - real_count),
            acting_seat=seat,
        )
