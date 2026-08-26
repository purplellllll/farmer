"""Policy registry and adapters for local Kaggriculture evaluation.

Built-ins are deliberately deterministic for a fixed match seed.  They are
evaluation opponents, not claims of leaderboard strength.  Public agents from
write-ups can be plugged in with ``python:path/to/file.py:agent`` without
copying their source into this repository.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import random
from typing import Any, Callable, Mapping, Sequence

from farmer_rl.actions import CROPS, PRODUCTS, SEED_COSTS, CandidateGenerator, JointActionCodec
from farmer_rl.environment import Action, pass_action
from farmer_rl.model import ModelConfig, build_actor_critic
from farmer_rl.tokenizer import ObservationTokenizer

Policy = Callable[[dict[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class BuiltinDescription:
    name: str
    family: str
    description: str


BUILTIN_POLICIES: dict[str, BuiltinDescription] = {
    "pass": BuiltinDescription("pass", "floor", "Always PASS; verifies scoring and the absolute floor."),
    "random": BuiltinDescription("random", "floor", "Seeded conservative random legal-candidate policy."),
    "starter": BuiltinDescription("starter", "official", "Official deterministic CARROT starter agent."),
    "crop_fast": BuiltinDescription("crop_fast", "production", "Short-cycle WHEAT/CARROT production with two hands."),
    "premium": BuiltinDescription("premium", "production", "Patient MELON route with late harvest and low labour."),
    "expansion": BuiltinDescription("expansion", "capital", "Mixed crops, early hiring, and affordable land expansion."),
    "market": BuiltinDescription("market", "market", "Selects crops from current public sale-price/seed-cost ratios."),
    "defense": BuiltinDescription("defense", "risk", "Low-capex CARROT route with early cash realization."),
}


def _own(observation: Mapping[str, Any]) -> tuple[int, Mapping[str, Any], Mapping[str, Any]]:
    seat = int(observation.get("player", -1))
    farms = observation.get("farms", [])
    if seat not in (0, 1) or not isinstance(farms, Sequence) or len(farms) != 2:
        raise ValueError("policy requires player=0/1 and exactly two farms")
    farm = farms[seat]
    private = observation.get("private", {}) or {}
    if not isinstance(farm, Mapping) or not isinstance(private, Mapping):
        raise ValueError("farm and private state must be mappings")
    return seat, farm, private


def _move_toward(position: Sequence[Any], target: tuple[int, int]) -> list[str]:
    x, y = int(position[0]), int(position[1])
    tx, ty = target
    if x < tx:
        return ["EAST"]
    if x > tx:
        return ["WEST"]
    if y < ty:
        return ["SOUTH"]
    if y > ty:
        return ["NORTH"]
    return ["PASS"]


def _distance(position: Sequence[Any], target: tuple[int, int]) -> int:
    return abs(int(position[0]) - target[0]) + abs(int(position[1]) - target[1])


class SeededRandomPolicy:
    """Random lower bound sampled only from the conservative legal set."""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.generator = CandidateGenerator(capacity=64)

    def __call__(self, observation: dict[str, Any]) -> Action:
        _, farm, _ = _own(observation)
        actions: list[list[Any]] = []
        for unit_index in range(1 + len(farm.get("hands", []) or [])):
            candidates = self.generator.unit_candidates(observation, unit_index).candidates
            actions.append(list(self.rng.choice(candidates).action))
        market: list[list[Any]] = []
        if self.rng.random() < 0.15:
            candidates = self.generator.market_candidates(observation, 0).candidates
            choice = list(self.rng.choice(candidates).action)
            if choice[0] != "NO_ORDER":
                market.append(choice)
        return {"farmer": actions[0], "hands": actions[1:], "market": market}


class FarmHeuristicPolicy:
    """A family of distinct, observation-only production/economic baselines."""

    def __init__(
        self,
        *,
        crops: tuple[str, ...],
        target_hands: int,
        expand_land: bool,
        harvest_mode: str,
        adaptive_market: bool = False,
    ) -> None:
        self.crops = crops
        self.target_hands = int(target_hands)
        self.expand_land = bool(expand_land)
        self.harvest_mode = harvest_mode
        self.adaptive_market = bool(adaptive_market)

    @staticmethod
    def _crop_constants() -> Mapping[str, Mapping[str, Any]]:
        # The pinned official engine is the source of truth when installed.
        try:
            from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS as official

            return official
        except ImportError:
            return {
                crop: {"seed": cost, "max_yield_day": day}
                for crop, cost, day in (
                    ("WHEAT", 10, 3),
                    ("CARROT", 20, 4),
                    ("TOMATO", 50, 5),
                    ("STRAWBERRY", 100, 7),
                    ("MELON", 80, 6),
                )
            }

    def _selected_crops(self, observation: Mapping[str, Any]) -> tuple[str, ...]:
        if not self.adaptive_market:
            return self.crops
        prices = ((observation.get("market", {}) or {}).get("prices", {}) or {})
        ranked = sorted(
            CROPS,
            key=lambda crop: float(prices.get(crop, 0.0) or 0.0) / max(1, SEED_COSTS[crop]),
            reverse=True,
        )
        return tuple(ranked[:3])

    def _unit_action(
        self,
        observation: Mapping[str, Any],
        unit_index: int,
        position: Sequence[Any],
        crops: tuple[str, ...],
    ) -> list[Any]:
        _, farm, private = _own(observation)
        board = farm.get("tiles", []) or []
        x, y = int(position[0]), int(position[1])
        tile = board[y][x] if 0 <= y < len(board) and 0 <= x < len(board[y]) else "LOCKED"
        inventories = private.get("inventories", []) or []
        carried = inventories[unit_index] if unit_index < len(inventories) else {}
        carried = carried if isinstance(carried, Mapping) else {}
        if any(int(value or 0) > 0 for value in carried.values()):
            half = len(board) // 2
            if (x, y) in {(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)}:
                return ["DROP"]

        if isinstance(tile, Mapping):
            kind = str(tile.get("kind", ""))
            if kind == "PLANT":
                planted = int(tile.get("planted_day", 0) or 0)
                age = int(observation.get("day", 0) or 0) - planted
                crop = str(tile.get("crop", "WHEAT"))
                max_day = int(self._crop_constants().get(crop, {}).get("max_yield_day", 4))
                yield_units = int(tile.get("yield_units", 0) or 0)
                ready = yield_units > 0 and (
                    self.harvest_mode == "early" or age >= max_day
                )
                if ready:
                    return ["HARVEST"]
                if not bool(tile.get("watered_today", False)):
                    return ["WATER"]
            elif kind == "WEED":
                return ["DIG"]
            elif kind in {"COOP", "PASTURE"} and int(tile.get("yield_units", 0) or 0) > 0:
                return ["HARVEST"]

        seeds = private.get("seeds", {}) or {}
        preferred = crops[unit_index % len(crops)]
        if tile is None and int(seeds.get(preferred, 0) or 0) > 0:
            return ["PLANT", preferred]

        targets: list[tuple[int, int, int]] = []
        for ty, row in enumerate(board):
            for tx, value in enumerate(row):
                priority = 9
                if isinstance(value, Mapping) and value.get("kind") == "PLANT":
                    if int(value.get("yield_units", 0) or 0) > 0:
                        priority = 0
                    elif not bool(value.get("watered_today", False)):
                        priority = 1
                elif value is None and int(seeds.get(preferred, 0) or 0) > 0:
                    priority = 3
                if priority < 9 and (tx, ty) != (x, y):
                    targets.append((priority, _distance(position, (tx, ty)), ty * len(row) + tx))
        if targets:
            _, _, flat = min(targets)
            width = len(board[0]) if board else 1
            return _move_toward(position, (flat % width, flat // width))
        return ["PASS"]

    def __call__(self, observation: dict[str, Any]) -> Action:
        _, farm, private = _own(observation)
        crops = self._selected_crops(observation)
        positions = [farm.get("farmer", [0, 0]), *(farm.get("hands", []) or [])]
        unit_actions = [
            self._unit_action(observation, index, position, crops)
            for index, position in enumerate(positions)
        ]

        market: list[list[Any]] = []
        shed = private.get("shed", {}) or {}
        for product in PRODUCTS:
            quantity = int(shed.get(product, 0) or 0)
            if quantity > 0:
                market.append(["SELL", product, quantity])
        money = int(float(farm.get("money", 0) or 0))
        committed = 0
        seeds = private.get("seeds", {}) or {}
        desired_seeds = max(2, len(positions) * 2)
        for crop in crops:
            missing = max(0, desired_seeds - int(seeds.get(crop, 0) or 0))
            if missing and committed + SEED_COSTS[crop] <= money and len(market) < 10:
                quantity = min(missing, max(1, (money - committed) // SEED_COSTS[crop]))
                market.append(["BUY_SEED", crop, quantity])
                committed += quantity * SEED_COSTS[crop]
        day = int(observation.get("day", 0) or 0)
        hands = len(farm.get("hands", []) or [])
        if hands < self.target_hands and day < 12 and len(market) < 10:
            hires_today = int(farm.get("hires_today", 0) or 0)
            a, b = 1, 1
            for _ in range(max(0, hires_today)):
                a, b = b, a + b
            if committed + a <= money:
                market.append(["HIRE"])
                committed += a
        quadrants = farm.get("unlocked_quadrants", []) or ["NW"]
        if self.expand_land and len(quadrants) < 4 and day < 18 and len(market) < 10:
            cost = (1000, 2000, 4000)[min(len(quadrants) - 1, 2)]
            if committed + cost <= money:
                market.append(["BUY_LAND"])
        return {"farmer": unit_actions[0], "hands": unit_actions[1:], "market": market[:10]}


class CheckpointPolicy:
    """Deterministic CPU-by-default actor for BC and native-PPO checkpoints."""

    def __init__(self, path: str | Path, *, device: str = "cpu") -> None:
        import torch

        resolved = _resolve_checkpoint_path(Path(path))
        payload = torch.load(resolved, map_location="cpu", weights_only=False)
        checkpoint_format = payload.get("format")
        if checkpoint_format == "farmer-rl-bc/v1":
            state_dict = payload["state_dict"]
        elif checkpoint_format == "farmer-native-ppo/v1":
            state_dict = payload["model"]
        else:
            raise ValueError(f"unsupported checkpoint format: {checkpoint_format!r}")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA checkpoint evaluation requested but CUDA is unavailable")
        self.config = ModelConfig.from_dict(dict(payload["model_config"]))
        self.model = build_actor_critic(self.config).to(self.device)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.tokenizer = ObservationTokenizer(max_tokens=self.config.max_tokens)
        self.codec = JointActionCodec(
            CandidateGenerator(capacity=self.config.candidate_capacity),
            max_hands=self.config.slots - 11,
            max_orders=10,
        )

    def __call__(self, observation: dict[str, Any]) -> Action:
        import torch

        tokenized = self.tokenizer.tokenize(observation)
        mask = self.codec.mask(observation)
        with torch.inference_mode():
            logits, _ = self.model(
                torch.tensor(tokenized.values, dtype=torch.float32, device=self.device).unsqueeze(0),
                torch.tensor(tokenized.token_type_ids, dtype=torch.long, device=self.device).unsqueeze(0),
                torch.tensor(tokenized.attention_mask, dtype=torch.bool, device=self.device).unsqueeze(0),
                torch.tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0),
            )
        indices = logits.argmax(dim=-1).squeeze(0).cpu().tolist()
        return self.codec.decode(observation, indices)


def _resolve_checkpoint_path(path: Path) -> Path:
    if path.is_dir():
        latest = path / "latest.json"
        if not latest.exists():
            latest = path.parent / "latest.json"
        if not latest.exists():
            raise FileNotFoundError(f"checkpoint directory has no latest.json: {path}")
        path = latest
    if path.suffix.lower() == ".json":
        import json

        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        checkpoint = Path(value["checkpoint"])
        path = checkpoint if checkpoint.is_absolute() else path.parent / checkpoint
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def _external_policy(value: str) -> Policy:
    file_value, separator, attribute = value.rpartition(":")
    if not separator or not file_value or not attribute:
        raise ValueError("python policy syntax is python:path/to/agent.py:function")
    path = Path(file_value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    module_name = f"farmer_eval_external_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import external policy: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    policy = getattr(module, attribute)
    if not callable(policy):
        raise TypeError(f"{attribute!r} in {path} is not callable")
    return policy


def load_policy(spec: str, *, seed: int = 0) -> Policy:
    """Resolve a reproducible built-in, external Python agent, or RL checkpoint."""

    if spec.startswith("builtin:"):
        name = spec.partition(":")[2]
    elif spec in BUILTIN_POLICIES:
        name = spec
    else:
        name = ""
    if name:
        if name not in BUILTIN_POLICIES:
            raise KeyError(f"unknown built-in policy {name!r}; choices={sorted(BUILTIN_POLICIES)}")
        if name == "pass":
            return lambda observation: pass_action(len(_own(observation)[1].get("hands", []) or []))
        if name == "random":
            return SeededRandomPolicy(seed)
        if name == "starter":
            from kaggle_environments.envs.kaggriculture.kaggriculture import starter_agent

            return starter_agent
        if name == "crop_fast":
            return FarmHeuristicPolicy(crops=("WHEAT", "CARROT"), target_hands=2, expand_land=False, harvest_mode="mature")
        if name == "premium":
            return FarmHeuristicPolicy(crops=("MELON",), target_hands=1, expand_land=False, harvest_mode="mature")
        if name == "expansion":
            return FarmHeuristicPolicy(crops=("WHEAT", "CARROT", "TOMATO"), target_hands=4, expand_land=True, harvest_mode="mature")
        if name == "market":
            return FarmHeuristicPolicy(crops=("CARROT",), target_hands=2, expand_land=False, harvest_mode="mature", adaptive_market=True)
        if name == "defense":
            return FarmHeuristicPolicy(crops=("CARROT",), target_hands=0, expand_land=False, harvest_mode="early")
    if spec.startswith("python:"):
        return _external_policy(spec[len("python:") :])
    if spec.startswith("checkpoint:"):
        value = spec[len("checkpoint:") :]
        path, marker, device = value.rpartition("@")
        if marker and device in {"cpu", "cuda"}:
            return CheckpointPolicy(path, device=device)
        return CheckpointPolicy(value, device="cpu")
    raise ValueError(
        "policy must be a built-in name, builtin:name, python:path.py:function, "
        "or checkpoint:path[@cpu|@cuda]"
    )
