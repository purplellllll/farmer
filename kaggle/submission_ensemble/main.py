"""Kaggriculture eight-model farmer router submission.

The ensemble selects the bounded farmer route.  Arguments and market orders are
supplied by a deterministic carrot controller, which also serves as a fail-safe
if a model dependency cannot be loaded in the remote agent image.
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Any, Mapping

# The competition simulator uses a leaner Python 3.11 image than Kaggle
# notebooks.  Prefer the ABI-matched wheels vendored beside this file before
# importing NumPy or any estimator package.
_AGENT_ROOT = "/kaggle_simulations/agent"
if os.path.isdir(_AGENT_ROOT):
    if _AGENT_ROOT in sys.path:
        sys.path.remove(_AGENT_ROOT)
    sys.path.insert(0, _AGENT_ROOT)
for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import numpy as np


_BUNDLE = None
_LOAD_ATTEMPTED = False
_PREDICT_CONFIRMED = False
_PREDICT_ERROR_LOGGED = False

_CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
_PRODUCTS = (
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "FERTILIZER",
)
_RESOURCES = tuple(dict.fromkeys((*_CROPS, *_PRODUCTS)))
_RESOURCE_IDS = {name: index + 1 for index, name in enumerate(_RESOURCES)}
_TILE_IDS = {"EMPTY": 0, "LOCKED": 1, "PLANT": 2, "WEED": 3, "COOP": 4, "PASTURE": 5}


def _number(value: Any, scale: float = 1.0) -> float:
    try:
        return float(value or 0.0) / scale
    except (TypeError, ValueError):
        return 0.0


def _features(observation: Mapping[str, Any]) -> np.ndarray:
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
        _number(_TILE_IDS.get(tile_kind, 0), max(_TILE_IDS.values())),
        _number(_RESOURCE_IDS.get(resource, 0), max(1, len(_RESOURCE_IDS))),
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
    values.extend(_number(seeds.get(name), 100.0) for name in _CROPS)
    values.extend(_number(shed.get(name), 100.0) for name in _PRODUCTS)
    values.extend(_number(prices.get(name), 500.0) for name in _PRODUCTS)
    values.extend(_number(inventory.get(name), 10_000.0) for name in _PRODUCTS)
    return np.asarray(values, dtype=np.float32)


def _submission_dir(configuration: Mapping[str, Any]) -> str:
    candidates = ["/kaggle_simulations/agent"]
    raw_path = configuration.get("__raw_path__")
    if raw_path and len(str(raw_path)) < 4096:
        candidates.append(os.path.dirname(os.path.abspath(str(raw_path))))
    file_path = globals().get("__file__")
    if file_path:
        candidates.append(os.path.dirname(os.path.abspath(str(file_path))))
    candidates.append(os.getcwd())
    for candidate in candidates:
        try:
            if os.path.isfile(os.path.join(candidate, "ensemble_bundle.pkl")):
                return candidate
        except OSError:
            continue
    return candidates[0]


def _force_cpu(bundle: Any) -> None:
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


def _get_bundle(configuration: Mapping[str, Any]) -> Any:
    global _BUNDLE, _LOAD_ATTEMPTED
    if _BUNDLE is not None or _LOAD_ATTEMPTED:
        return _BUNDLE
    _LOAD_ATTEMPTED = True
    base = _submission_dir(configuration)
    if base not in sys.path:
        sys.path.insert(0, base)
    try:
        from farmer_ensemble.ensemble import load_bundle

        _BUNDLE = load_bundle(os.path.join(base, "ensemble_bundle.pkl"), verify_manifest=False)
        _force_cpu(_BUNDLE)
    except Exception:
        # The deterministic policy remains a valid agent if a remote package is
        # unexpectedly unavailable.  Avoid logging 720 repeated tracebacks.
        traceback.print_exc()
        _BUNDLE = None
    return _BUNDLE


def _controller(observation: Mapping[str, Any], route: int | None) -> dict[str, Any]:
    farms = observation.get("farms", [])
    player = int(observation.get("player", 0))
    private = observation.get("private", {}) or {}
    if not farms or player >= len(farms):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    farm = farms[player]
    x, y = farm.get("farmer", [0, 0])
    tile = farm.get("tiles", [])[int(y)][int(x)]
    seeds = private.get("seeds", {}) or {}
    shed = private.get("shed", {}) or {}
    market = []
    carrots = int(shed.get("CARROT", 0) or 0)
    if carrots > 0:
        market.append(["SELL", "CARROT", carrots])
    if int(seeds.get("CARROT", 0) or 0) == 0 and float(farm.get("money", 0) or 0) >= 20:
        market.append(["BUY_SEED", "CARROT", 1])

    fallback = ["PASS"]
    if tile is None and int(seeds.get("CARROT", 0) or 0) > 0:
        fallback = ["PLANT", "CARROT"]
    elif isinstance(tile, Mapping) and tile.get("kind") == "PLANT" and tile.get("crop") == "CARROT":
        age = int(observation.get("day", 0)) - int(tile.get("planted_day", 0))
        if age >= 3:
            fallback = ["HARVEST"]
        elif not bool(tile.get("watered_today", False)):
            fallback = ["WATER"]

    farmer = fallback
    if route == 0:
        farmer = ["PASS"]
    elif route == 5 and tile is None and int(seeds.get("CARROT", 0) or 0) > 0:
        farmer = ["PLANT", "CARROT"]
    elif route == 6 and isinstance(tile, Mapping) and tile.get("kind") == "PLANT":
        farmer = ["WATER"]
    elif route == 7 and isinstance(tile, Mapping) and tile.get("kind") == "PLANT":
        farmer = ["HARVEST"]

    hands = [["PASS"] for _ in (farm.get("hands", []) or [])]
    return {"farmer": farmer, "hands": hands, "market": market}


def agent(observation: Mapping[str, Any], configuration: Mapping[str, Any]) -> dict[str, Any]:
    global _PREDICT_CONFIRMED, _PREDICT_ERROR_LOGGED
    route = None
    bundle = _get_bundle(configuration)
    if bundle is not None:
        try:
            route = int(bundle.predict(_features(observation)[None, :])[0])
            if not _PREDICT_CONFIRMED:
                print(f"ENSEMBLE_RUNTIME_OK models={len(bundle.base_models)} route={route}")
                _PREDICT_CONFIRMED = True
        except Exception:
            if not _PREDICT_ERROR_LOGGED:
                traceback.print_exc()
                _PREDICT_ERROR_LOGGED = True
            route = None
    return _controller(observation, route)
