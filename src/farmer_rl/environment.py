"""A small, acting-seat-safe wrapper around the official Kaggle environment.

This is intentionally not a Gym/RLlib adapter.  It is the stable core contract;
``farmer_rl.rllib_entry`` adds framework-specific spaces without contaminating
the data collection layer with optional dependencies.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .errors import InvalidActionError, OptionalDependencyError, SeatSafetyError

Action = dict[str, Any]
Observation = dict[str, Any]


@dataclass(frozen=True)
class StepResult:
    observations: dict[int, Observation]
    rewards: dict[int, float]
    terminated: bool
    truncated: bool
    statuses: dict[int, str]
    info: dict[str, Any]


def pass_action(hand_count: int = 0) -> Action:
    """Return a JSON-safe no-op action with exactly one action per current hand."""

    return {
        "farmer": ["PASS"],
        "hands": [["PASS"] for _ in range(max(0, int(hand_count)))],
        "market": [],
    }


def validate_action(action: Mapping[str, Any], *, hand_count: int | None = None) -> Action:
    """Validate only the public action envelope, not game-state legality.

    State-dependent legality belongs to :mod:`farmer_rl.actions`.  Separating
    envelope validation from legality is important because the official engine
    treats illegal operations as silent no-ops.
    """

    if not isinstance(action, Mapping):
        raise InvalidActionError("action must be a mapping")
    if set(action) != {"farmer", "hands", "market"}:
        raise InvalidActionError("action keys must be exactly farmer, hands, market")
    farmer = action["farmer"]
    hands = action["hands"]
    market = action["market"]
    if not isinstance(farmer, (list, tuple)) or not farmer or not isinstance(farmer[0], str):
        raise InvalidActionError("farmer action must be a non-empty operation sequence")
    if not isinstance(hands, (list, tuple)) or any(
        not isinstance(item, (list, tuple)) or not item or not isinstance(item[0], str)
        for item in hands
    ):
        raise InvalidActionError("hands must contain one non-empty operation sequence per hand")
    if hand_count is not None and len(hands) != hand_count:
        raise InvalidActionError(f"expected {hand_count} hand actions, got {len(hands)}")
    if not isinstance(market, (list, tuple)) or len(market) > 10 or any(
        not isinstance(item, (list, tuple)) or not item or not isinstance(item[0], str)
        for item in market
    ):
        raise InvalidActionError("market must contain at most ten operation sequences")
    # Round-trip through lists and deepcopy to prevent policies mutating recorded actions.
    return {
        "farmer": list(farmer),
        "hands": [list(item) for item in hands],
        "market": [list(item) for item in market],
    }


class KaggricultureEnv:
    """Two-seat stepping interface backed by the official environment.

    ``make_fn`` is injectable so unit tests require neither network access nor
    ``kaggle-environments``.  Production callers should pin the package version
    recorded in ``docs/rl/sources.md``.
    """

    def __init__(
        self,
        *,
        configuration: Mapping[str, Any] | None = None,
        debug: bool = False,
        make_fn: Callable[..., Any] | None = None,
        environment_version: str = "kaggle-environments==1.32.7",
    ) -> None:
        self.configuration = {"episodeSteps": 720, **dict(configuration or {})}
        self.debug = bool(debug)
        self._make_fn = make_fn
        self.environment_version = environment_version
        self._env: Any | None = None
        self._last_observations: dict[int, Observation] = {}

    @staticmethod
    def _official_make() -> Callable[..., Any]:
        try:
            from kaggle_environments import make
        except ImportError as exc:  # pragma: no cover - exercised without optional package
            raise OptionalDependencyError(
                "Official simulation requested but kaggle-environments is missing. "
                "Install the pinned package: pip install kaggle-environments==1.32.7"
            ) from exc
        return make

    @staticmethod
    def _field(state: Any, name: str, default: Any = None) -> Any:
        if isinstance(state, Mapping):
            return state.get(name, default)
        return getattr(state, name, default)

    @classmethod
    def _seat_observations(cls, states: Sequence[Any]) -> dict[int, Observation]:
        if len(states) != 2:
            raise SeatSafetyError(f"Kaggriculture requires exactly two seats, got {len(states)}")
        result: dict[int, Observation] = {}
        for seat, state in enumerate(states):
            raw = cls._field(state, "observation", state)
            if not isinstance(raw, Mapping):
                raise SeatSafetyError(f"seat {seat} observation is not a mapping")
            observation = deepcopy(dict(raw))
            observed_seat = observation.get("player")
            if observed_seat is None:
                # Some test doubles omit framework-added fields. Adding the known
                # acting seat is safe; overwriting a conflicting value is not.
                observation["player"] = seat
            elif int(observed_seat) != seat:
                raise SeatSafetyError(
                    f"state index {seat} carries player={observed_seat}; refusing cross-seat label"
                )
            result[seat] = observation
        return result

    def reset(self, *, seed: int | None = None) -> dict[int, Observation]:
        make_fn = self._make_fn or self._official_make()
        config = dict(self.configuration)
        if seed is not None:
            config["seed"] = int(seed)
        self._env = make_fn("kaggriculture", configuration=config, debug=self.debug)
        states = self._env.reset(num_agents=2)
        self._last_observations = self._seat_observations(states)
        return deepcopy(self._last_observations)

    def step(self, actions: Mapping[int, Mapping[str, Any]]) -> StepResult:
        if self._env is None:
            raise RuntimeError("reset() must be called before step()")
        if set(actions) != {0, 1}:
            raise InvalidActionError("simultaneous step requires actions for seats 0 and 1")
        validated: list[Action] = []
        for seat in (0, 1):
            obs = self._last_observations[seat]
            farms = obs.get("farms", [])
            hand_count = 0
            if isinstance(farms, Sequence) and len(farms) > seat and isinstance(farms[seat], Mapping):
                hand_count = len(farms[seat].get("hands", []) or [])
            validated.append(validate_action(actions[seat], hand_count=hand_count))

        states = self._env.step(validated)
        observations = self._seat_observations(states)
        rewards = {
            seat: float(self._field(state, "reward", 0.0) or 0.0)
            for seat, state in enumerate(states)
        }
        statuses = {
            seat: str(self._field(state, "status", "ACTIVE"))
            for seat, state in enumerate(states)
        }
        terminated = all(status in {"DONE", "INVALID", "ERROR"} for status in statuses.values())
        self._last_observations = observations
        return StepResult(
            observations=deepcopy(observations),
            rewards=rewards,
            terminated=terminated,
            truncated=False,
            statuses=statuses,
            info={"environment_version": self.environment_version},
        )

    @property
    def raw_environment(self) -> Any:
        """Expose the official object for rendering only; never use it as actor input."""

        if self._env is None:
            raise RuntimeError("environment has not been reset")
        return self._env
