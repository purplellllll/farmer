"""Configuration-driven frozen-opponent pool with PFSP-style sampling."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class OpponentSpec:
    opponent_id: str
    kind: str
    weight: float = 1.0
    enabled: bool = True
    artifact: str | None = None
    source_id: str | None = None
    frozen: bool = True
    tags: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OpponentSpec":
        data = dict(value)
        data["tags"] = tuple(data.get("tags", ()))
        return cls(**data)


@dataclass
class OpponentStats:
    games: int = 0
    learner_wins: float = 0.0

    @property
    def learner_win_rate(self) -> float:
        return self.learner_wins / self.games if self.games else 0.5


@dataclass
class OpponentPool:
    opponents: tuple[OpponentSpec, ...]
    pfsp_power: float = 1.0
    stats: dict[str, OpponentStats] = field(default_factory=dict)

    def __post_init__(self) -> None:
        enabled = [item for item in self.opponents if item.enabled]
        if not enabled:
            raise ValueError("opponent pool has no enabled opponent")
        for item in self.opponents:
            if item.weight < 0:
                raise ValueError("opponent weights must be non-negative")
            self.stats.setdefault(item.opponent_id, OpponentStats())

    @classmethod
    def from_json(cls, path: str | Path) -> "OpponentPool":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(
            opponents=tuple(OpponentSpec.from_dict(item) for item in payload["opponents"]),
            pfsp_power=float(payload.get("pfsp_power", 1.0)),
        )

    def sampling_weights(self) -> dict[str, float]:
        """Prioritise near-even opponents while retaining explicit base weights."""

        result: dict[str, float] = {}
        for opponent in self.opponents:
            if not opponent.enabled:
                continue
            win_rate = self.stats[opponent.opponent_id].learner_win_rate
            # Maximum at 50%; a small floor prevents catastrophic forgetting.
            pfsp = max(0.05, 1.0 - abs(2.0 * win_rate - 1.0)) ** self.pfsp_power
            result[opponent.opponent_id] = opponent.weight * pfsp
        return result

    def sample(self, rng: random.Random | None = None) -> OpponentSpec:
        rng = rng or random.Random()
        enabled = [item for item in self.opponents if item.enabled]
        by_id = self.sampling_weights()
        weights = [by_id[item.opponent_id] for item in enabled]
        if sum(weights) <= 0:
            weights = [1.0] * len(enabled)
        return rng.choices(enabled, weights=weights, k=1)[0]

    def record(self, opponent_id: str, outcome: float) -> None:
        """Record learner outcome as 1 win, .5 draw, or 0 loss."""

        if opponent_id not in self.stats:
            raise KeyError(opponent_id)
        if outcome not in (0.0, 0.5, 1.0):
            raise ValueError("outcome must be 0, 0.5, or 1")
        stats = self.stats[opponent_id]
        stats.games += 1
        stats.learner_wins += outcome
