"""Leakage-safe decision-router data for complete Kaggriculture actions.

The first router dataset contained one label per turn: the main farmer's
operation.  It consequently could not learn hand actions or market orders.
This module emits one row per *active decision slot* instead:

* one row for the main farmer and each active farm hand;
* one row for every submitted market order; and
* one ``NO_ORDER`` row terminating the market queue.

The terminating representation is deliberate.  Padding every turn to ten
independently classified market rows taught early policies to repeat the same
purchase ten times.  A single terminator plus a stateful decoder lets inference
reserve cash/resources across the ordered queue.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from farmer_rl.actions import ANIMALS, CROPS, PRODUCTS

from .router_dataset import (
    RESOURCE_IDS,
    RESOURCE_NAMES,
    ROUTE_NAMES,
    TILE_IDS,
    extract_features,
    feature_names,
)


MARKET_ROUTE_NAMES = (
    "NO_ORDER",
    "BUY_SEED",
    "BUY_ANIMAL",
    "BUY_PRODUCT",
    "SELL",
    "HIRE",
    "BUY_LAND",
)
DECISION_NAMES = (*ROUTE_NAMES, *MARKET_ROUTE_NAMES)
DECISION_IDS = {name: index for index, name in enumerate(DECISION_NAMES)}
SLOT_NAMES = ("farmer", "hand", "market")
INVENTORY_NAMES = tuple(dict.fromkeys((*RESOURCE_NAMES, *ANIMALS)))


def decision_feature_names() -> list[str]:
    names = feature_names()
    names.extend(
        (
            "slot_is_farmer",
            "slot_is_hand",
            "slot_is_market",
            "unit_index",
            "market_order_index",
            "unit_x",
            "unit_y",
            "unit_tile_kind",
            "unit_tile_resource",
            "unit_tile_age",
            "unit_tile_yield",
            "unit_watered_today",
            "unit_fed_today",
            "unit_cared_today",
            "unit_consecutive_unwatered",
            "unit_consecutive_unfed",
        )
    )
    names.extend(f"unit_inventory_{name.lower()}" for name in INVENTORY_NAMES)
    return names


def _number(value: Any, scale: float = 1.0) -> float:
    try:
        return float(value or 0.0) / scale
    except (TypeError, ValueError):
        return 0.0


def extract_decision_features(
    observation: Mapping[str, Any],
    *,
    slot: str,
    unit_index: int = -1,
    market_order_index: int = -1,
) -> np.ndarray:
    """Extract deployable features for a unit or ordered-market decision."""

    if slot not in SLOT_NAMES:
        raise ValueError(f"unknown decision slot {slot!r}")
    base = extract_features(observation).tolist()
    seat = int(observation.get("player", 0))
    farm = (observation.get("farms", []) or [])[seat]
    board = farm.get("tiles", []) or []
    height = len(board)
    width = len(board[0]) if height else 0
    private = observation.get("private", {}) or {}
    inventories = private.get("inventories", []) or []

    tile: Any = None
    inventory: Mapping[str, Any] = {}
    x = y = 0
    if slot != "market":
        positions = [farm.get("farmer", [0, 0]), *(farm.get("hands", []) or [])]
        if not 0 <= unit_index < len(positions):
            raise IndexError(f"unit_index {unit_index} is not active")
        x, y = (int(value) for value in positions[unit_index])
        tile = board[y][x] if 0 <= y < height and 0 <= x < width else None
        if unit_index < len(inventories) and isinstance(inventories[unit_index], Mapping):
            inventory = inventories[unit_index]

    if tile is None:
        tile_kind = "EMPTY"
        tile_data: Mapping[str, Any] = {}
    elif tile == "LOCKED":
        tile_kind = "LOCKED"
        tile_data = {}
    elif isinstance(tile, Mapping):
        tile_kind = str(tile.get("kind", "EMPTY"))
        tile_data = tile
    else:
        tile_kind = "EMPTY"
        tile_data = {}
    resource = str(tile_data.get("crop") or tile_data.get("animal") or "")
    planted = tile_data.get(
        "planted_day", tile_data.get("placed_day", observation.get("day", 0))
    )
    context = [
        float(slot == "farmer"),
        float(slot == "hand"),
        float(slot == "market"),
        _number(max(unit_index, 0), 16.0) if slot != "market" else -1.0,
        _number(max(market_order_index, 0), 9.0) if slot == "market" else -1.0,
        _number(x, max(1, width - 1)),
        _number(y, max(1, height - 1)),
        _number(TILE_IDS.get(tile_kind, 0), max(TILE_IDS.values())),
        _number(RESOURCE_IDS.get(resource, 0), max(1, len(RESOURCE_IDS))),
        _number(float(observation.get("day", 0)) - float(planted or 0), 30.0),
        _number(tile_data.get("yield_units"), 10.0),
        float(bool(tile_data.get("watered_today", False))),
        float(bool(tile_data.get("fed_today", False))),
        float(bool(tile_data.get("cared_today", False))),
        _number(tile_data.get("consecutive_unwatered"), 2.0),
        _number(tile_data.get("consecutive_unfed"), 2.0),
    ]
    context.extend(_number(inventory.get(name), 100.0) for name in INVENTORY_NAMES)
    return np.asarray((*base, *context), dtype=np.float32)


def _operation(action: Any) -> str:
    if not isinstance(action, (list, tuple)) or not action:
        raise ValueError(f"invalid operation sequence: {action!r}")
    return str(action[0])


def record_to_rows(record: Mapping[str, Any]) -> list[tuple[np.ndarray, int, str]]:
    """Convert one acting-seat transition into ordered decision rows."""

    observation = record["observation"]
    action = record["action"]
    farmer = _operation(action.get("farmer", ["PASS"]))
    if farmer not in DECISION_IDS:
        raise ValueError(f"unsupported farmer operation {farmer!r}")
    rows = [
        (
            extract_decision_features(observation, slot="farmer", unit_index=0),
            DECISION_IDS[farmer],
            "farmer",
        )
    ]
    hands = list(action.get("hands", []) or [])
    active_hands = len(
        (observation.get("farms", []) or [])[int(observation.get("player", 0))].get(
            "hands", []
        )
        or []
    )
    if len(hands) != active_hands:
        raise ValueError(f"expected {active_hands} hand actions, got {len(hands)}")
    for hand_index, hand_action in enumerate(hands):
        operation = _operation(hand_action)
        if operation not in DECISION_IDS:
            raise ValueError(f"unsupported hand operation {operation!r}")
        rows.append(
            (
                extract_decision_features(
                    observation, slot="hand", unit_index=hand_index + 1
                ),
                DECISION_IDS[operation],
                "hand",
            )
        )

    market = list(action.get("market", []) or [])
    if len(market) > 10:
        raise ValueError("market queue exceeds ten orders")
    seen: set[tuple[Any, ...]] = set()
    for order_index, order in enumerate(market):
        key = tuple(order)
        if key in seen:
            raise ValueError(f"duplicate market order in one joint action: {order!r}")
        seen.add(key)
        operation = _operation(order)
        if operation not in DECISION_IDS or operation == "NO_ORDER":
            raise ValueError(f"unsupported submitted market operation {operation!r}")
        rows.append(
            (
                extract_decision_features(
                    observation, slot="market", market_order_index=order_index
                ),
                DECISION_IDS[operation],
                "market",
            )
        )
    # One terminator models the queue length without nine or ten padded negatives.
    rows.append(
        (
            extract_decision_features(
                observation, slot="market", market_order_index=len(market)
            ),
            DECISION_IDS["NO_ORDER"],
            "market",
        )
    )
    return rows


def build_joint_router_npz(
    inputs: Iterable[str | Path], output: str | Path
) -> dict[str, Any]:
    rows: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    seeds: list[int] = []
    seats: list[int] = []
    slots: list[int] = []
    action_counts: Counter[str] = Counter()
    slot_counts: Counter[str] = Counter()
    duplicate_market_orders = 0
    transition_count = 0
    for input_value in inputs:
        with Path(input_value).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                transition_count += 1
                market = list(record.get("action", {}).get("market", []) or [])
                duplicate_market_orders += len(market) - len({tuple(order) for order in market})
                for features, label, slot in record_to_rows(record):
                    rows.append(features)
                    labels.append(label)
                    groups.append(str(record["episode_id"]))
                    seeds.append(int(record.get("seed") if record.get("seed") is not None else -1))
                    seats.append(int(record["acting_seat"]))
                    slots.append(SLOT_NAMES.index(slot))
                    action_counts[DECISION_NAMES[label]] += 1
                    slot_counts[slot] += 1
    if not rows:
        raise ValueError("no decision-router rows were produced")
    missing = sorted(set(DECISION_NAMES) - set(action_counts))
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=np.stack(rows),
        y=np.asarray(labels, dtype=np.int64),
        groups=np.asarray(groups),
        seeds=np.asarray(seeds, dtype=np.int64),
        seats=np.asarray(seats, dtype=np.int8),
        slots=np.asarray(slots, dtype=np.int8),
        feature_names=np.asarray(decision_feature_names()),
        class_names=np.asarray(DECISION_NAMES),
    )
    return {
        "schema_version": "kaggriculture-joint-router/v2",
        "transitions": transition_count,
        "rows": len(rows),
        "features": len(rows[0]),
        "classes": dict(sorted(action_counts.items())),
        "missing_classes": missing,
        "slot_counts": dict(sorted(slot_counts.items())),
        "duplicate_market_orders": duplicate_market_orders,
        "market_representation": "submitted orders followed by one NO_ORDER terminator",
        "output": str(path),
    }
