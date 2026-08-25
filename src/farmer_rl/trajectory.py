"""Versioned, acting-seat-safe trajectory records.

The schema stores exactly the observation delivered to the acting policy.  It
does not accept a simulator-global state, an opponent private state, or future
town unlocks.  This prevents an easy-to-miss train/deploy leakage bug.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from .environment import validate_action
from .errors import SeatSafetyError

SCHEMA_VERSION = "kaggriculture-trajectory/v1"


def _validate_seat(observation: Mapping[str, Any], seat: int, label: str) -> None:
    if not isinstance(observation, Mapping):
        raise SeatSafetyError(f"{label} must be a mapping")
    observed = observation.get("player")
    if observed is None or int(observed) != int(seat):
        raise SeatSafetyError(f"{label}.player={observed!r}, expected acting seat {seat}")


@dataclass(frozen=True)
class Transition:
    episode_id: str
    step: int
    acting_seat: int
    observation: dict[str, Any]
    action: dict[str, Any]
    reward: float
    next_observation: dict[str, Any]
    terminated: bool
    opponent_id: str
    policy_id: str
    seed: int | None = None
    environment_version: str = "kaggle-environments==1.32.7"
    source_id: str = "self-generated"
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema version: {self.schema_version}")
        if self.acting_seat not in (0, 1):
            raise SeatSafetyError("acting_seat must be 0 or 1")
        if self.step < 0:
            raise ValueError("step must be non-negative")
        _validate_seat(self.observation, self.acting_seat, "observation")
        _validate_seat(self.next_observation, self.acting_seat, "next_observation")
        validate_action(self.action)
        if not self.episode_id or not self.policy_id or not self.opponent_id:
            raise ValueError("episode_id, policy_id and opponent_id are required")
        # Frozen prevents reassignment, while deep copies prevent nested mutation.
        object.__setattr__(self, "observation", deepcopy(self.observation))
        object.__setattr__(self, "next_observation", deepcopy(self.next_observation))
        object.__setattr__(self, "action", deepcopy(self.action))
        object.__setattr__(self, "metadata", deepcopy(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Transition":
        return cls(**dict(value))


@dataclass
class EpisodeTrajectory:
    episode_id: str
    transitions: list[Transition] = field(default_factory=list)

    def append(self, transition: Transition) -> None:
        if transition.episode_id != self.episode_id:
            raise ValueError("transition episode_id does not match trajectory")
        previous = next(
            (item for item in reversed(self.transitions) if item.acting_seat == transition.acting_seat),
            None,
        )
        if previous is not None and transition.step <= previous.step:
            raise ValueError("per-seat trajectory steps must increase")
        self.transitions.append(transition)

    def for_seat(self, seat: int) -> list[Transition]:
        return [item for item in self.transitions if item.acting_seat == seat]

    def to_records(self) -> list[dict[str, Any]]:
        """Return memory-only records; callers choose storage and access policy."""

        return [item.to_dict() for item in self.transitions]
