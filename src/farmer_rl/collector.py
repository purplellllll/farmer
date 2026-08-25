"""In-memory collection for licensed local policies and the official simulator."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Mapping
from uuid import uuid4

from .environment import KaggricultureEnv, validate_action
from .trajectory import EpisodeTrajectory, Transition

Policy = Callable[[dict[str, Any]], Mapping[str, Any]]


def collect_episode(
    env: KaggricultureEnv,
    policies: Mapping[int, Policy],
    *,
    policy_ids: Mapping[int, str],
    seed: int | None = None,
    episode_id: str | None = None,
    source_id: str = "self-generated",
    max_steps: int | None = None,
) -> EpisodeTrajectory:
    """Collect a closed-loop episode without writing replay or raw data files.

    Each policy is called only with its own defensive-copy observation.  The
    caller must choose policies whose source is allowlisted by the data manifest.
    """

    if set(policies) != {0, 1} or set(policy_ids) != {0, 1}:
        raise ValueError("policies and policy_ids must contain exactly seats 0 and 1")
    observations = env.reset(seed=seed)
    trajectory = EpisodeTrajectory(episode_id or uuid4().hex)
    limit = int(max_steps or env.configuration.get("episodeSteps", 720))

    for step in range(limit):
        actions: dict[int, dict[str, Any]] = {}
        for seat in (0, 1):
            action = policies[seat](deepcopy(observations[seat]))
            farms = observations[seat].get("farms", [])
            hand_count = len(farms[seat].get("hands", []) or []) if len(farms) > seat else 0
            actions[seat] = validate_action(action, hand_count=hand_count)
        result = env.step(actions)
        for seat in (0, 1):
            trajectory.append(
                Transition(
                    episode_id=trajectory.episode_id,
                    step=step,
                    acting_seat=seat,
                    observation=observations[seat],
                    action=actions[seat],
                    reward=result.rewards[seat],
                    next_observation=result.observations[seat],
                    terminated=result.terminated,
                    opponent_id=policy_ids[1 - seat],
                    policy_id=policy_ids[seat],
                    seed=seed,
                    environment_version=env.environment_version,
                    source_id=source_id,
                    metadata={"status": result.statuses[seat]},
                )
            )
        observations = result.observations
        if result.terminated or result.truncated:
            break
    return trajectory
