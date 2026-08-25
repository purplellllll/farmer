"""Build a compact expert-router table from licensed trajectory JSONL."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from farmer_rl.actions import CROPS, PRODUCTS


ROUTE_NAMES = (
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
)
ROUTE_IDS = {name: index for index, name in enumerate(ROUTE_NAMES)}
TILE_IDS = {"EMPTY": 0, "LOCKED": 1, "PLANT": 2, "WEED": 3, "COOP": 4, "PASTURE": 5}
RESOURCE_NAMES = tuple(dict.fromkeys((*CROPS, *PRODUCTS)))
RESOURCE_IDS = {name: index + 1 for index, name in enumerate(RESOURCE_NAMES)}


def _number(value: Any, scale: float = 1.0) -> float:
    try:
        return float(value or 0.0) / scale
    except (TypeError, ValueError):
        return 0.0


def feature_names() -> list[str]:
    names = [
        "step",
        "day",
        "hour",
        "seat",
        "own_money",
        "opponent_money",
        "money_difference",
        "own_hands",
        "opponent_hands",
        "hires_today",
        "unlocked_quadrants",
        "farmer_x",
        "farmer_y",
        "current_tile_kind",
        "current_tile_resource",
        "current_tile_age",
        "current_tile_yield",
        "watered_today",
        "fed_today",
        "cared_today",
        "consecutive_unwatered",
        "consecutive_unfed",
        "unlocked_shop_count",
    ]
    names.extend(f"seed_{name.lower()}" for name in CROPS)
    names.extend(f"shed_{name.lower()}" for name in PRODUCTS)
    names.extend(f"market_price_{name.lower()}" for name in PRODUCTS)
    names.extend(f"market_inventory_{name.lower()}" for name in PRODUCTS)
    return names


def extract_features(observation: Mapping[str, Any]) -> np.ndarray:
    seat = int(observation.get("player", 0))
    farms = observation.get("farms", [])
    own = farms[seat]
    opponent = farms[1 - seat]
    private = observation.get("private", {}) or {}
    market = observation.get("market", {}) or {}
    x, y = own.get("farmer", [0, 0])
    board = own.get("tiles", []) or []
    height = len(board)
    width = len(board[0]) if height else 0
    tile = board[int(y)][int(x)] if 0 <= int(y) < height and 0 <= int(x) < width else None
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
    planted = tile_data.get("planted_day", tile_data.get("placed_day", observation.get("day", 0)))
    own_money = _number(own.get("money"))
    opponent_money = _number(opponent.get("money"))
    values = [
        _number(observation.get("step"), 720.0),
        _number(observation.get("day"), 30.0),
        _number(observation.get("hour"), 24.0),
        float(seat),
        own_money / 200_000.0,
        opponent_money / 200_000.0,
        (own_money - opponent_money) / 200_000.0,
        _number(len(own.get("hands", []) or []), 16.0),
        _number(len(opponent.get("hands", []) or []), 16.0),
        _number(own.get("hires_today"), 16.0),
        _number(len(own.get("unlocked_quadrants", []) or []), 4.0),
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
        _number(len((observation.get("town", {}) or {}).get("unlocked_shops", []) or []), 8.0),
    ]
    seeds = private.get("seeds", {}) or {}
    shed = private.get("shed", {}) or {}
    prices = market.get("prices", {}) or {}
    inventory = market.get("inventory", {}) or {}
    values.extend(_number(seeds.get(name), 100.0) for name in CROPS)
    values.extend(_number(shed.get(name), 100.0) for name in PRODUCTS)
    values.extend(_number(prices.get(name), 500.0) for name in PRODUCTS)
    values.extend(_number(inventory.get(name), 10_000.0) for name in PRODUCTS)
    return np.asarray(values, dtype=np.float32)


def build_router_npz(
    inputs: Iterable[str | Path], output: str | Path
) -> dict[str, Any]:
    rows: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    seeds: list[int] = []
    seats: list[int] = []
    for input_value in inputs:
        with Path(input_value).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                operation = str((record.get("action", {}).get("farmer") or ["PASS"])[0])
                if operation not in ROUTE_IDS:
                    continue
                rows.append(extract_features(record["observation"]))
                labels.append(ROUTE_IDS[operation])
                groups.append(str(record["episode_id"]))
                seeds.append(int(record.get("seed") or -1))
                seats.append(int(record["acting_seat"]))
    if not rows:
        raise ValueError("no router rows were produced")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=np.stack(rows),
        y=np.asarray(labels, dtype=np.int64),
        groups=np.asarray(groups),
        seeds=np.asarray(seeds, dtype=np.int64),
        seats=np.asarray(seats, dtype=np.int8),
        feature_names=np.asarray(feature_names()),
    )
    unique, counts = np.unique(labels, return_counts=True)
    return {
        "rows": len(rows),
        "features": len(rows[0]),
        "classes": {ROUTE_NAMES[int(k)]: int(v) for k, v in zip(unique, counts)},
        "output": str(path),
    }
