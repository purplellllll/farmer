"""Single-process CPU/CUDA PPO for local Kaggriculture self-play.

This runner exists because Ray's worker bootstrap is unreliable with CUDA on
some Windows laptops.  It intentionally reuses the exact tokenizer, legal
candidate codec, and residual Transformer actor-critic used by the RLlib path.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Callable, Mapping

import numpy as np

from .actions import ANIMAL_COSTS, SEED_COSTS, CandidateGenerator, JointActionCodec
from .environment import KaggricultureEnv, validate_action
from .model import ModelConfig, build_actor_critic
from .tokenizer import ObservationTokenizer


def _torch() -> Any:
    import torch

    return torch


def _encoded(
    observation: Mapping[str, Any],
    tokenizer: ObservationTokenizer,
    codec: JointActionCodec,
) -> dict[str, np.ndarray]:
    batch = tokenizer.tokenize(observation)
    return {
        "tokens": np.asarray(batch.values, dtype=np.float32),
        "types": np.asarray(batch.token_type_ids, dtype=np.int64),
        "attention": np.asarray(batch.attention_mask, dtype=np.bool_),
        "action_mask": np.asarray(codec.mask(observation), dtype=np.bool_),
    }


def _sample(
    model: Any,
    encoded: Mapping[str, np.ndarray],
    *,
    device: Any,
    observation: Mapping[str, Any] | None = None,
    codec: JointActionCodec | None = None,
    deterministic: bool = False,
) -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
    torch = _torch()
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
    ):
        logits, value = model(
            torch.as_tensor(encoded["tokens"], device=device).unsqueeze(0),
            torch.as_tensor(encoded["types"], device=device).unsqueeze(0),
            torch.as_tensor(encoded["attention"], device=device).unsqueeze(0),
            torch.as_tensor(encoded["action_mask"], device=device).unsqueeze(0),
        )
    if observation is not None and codec is not None:
        def chooser(slot_index: int, _candidate_set: Any, mask: tuple[int, ...]) -> int:
            row_mask = torch.as_tensor(mask, dtype=torch.bool, device=device)
            row_logits = logits[0, slot_index].float().masked_fill(
                ~row_mask, torch.finfo(torch.float32).min
            )
            if deterministic:
                return int(row_logits.argmax().item())
            return int(torch.distributions.Categorical(logits=row_logits).sample().item())

        selected, dynamic_mask_flat = codec.select(observation, chooser)
        actions = torch.as_tensor(selected, dtype=torch.long, device=device).unsqueeze(0)
        dynamic_mask = torch.as_tensor(
            dynamic_mask_flat, dtype=torch.bool, device=device
        ).reshape(1, codec.slots, codec.generator.capacity)
        logits = logits.float().masked_fill(~dynamic_mask, torch.finfo(torch.float32).min)
        action_mask = dynamic_mask.squeeze(0).cpu().numpy().astype(np.bool_)
    else:
        actions = logits.argmax(dim=-1) if deterministic else torch.distributions.Categorical(
            logits=logits.float()
        ).sample()
        action_mask = np.asarray(encoded["action_mask"], dtype=np.bool_).reshape(
            actions.shape[-1], -1
        )
    distribution = torch.distributions.Categorical(logits=logits.float())
    policy_slot_mask = torch.as_tensor(
        action_mask.sum(axis=-1) > 1, dtype=torch.float32, device=device
    ).unsqueeze(0)
    normalizer = policy_slot_mask.sum(dim=-1).clamp_min(1.0)
    log_probability = (
        distribution.log_prob(actions) * policy_slot_mask
    ).sum(dim=-1) / normalizer
    return (
        actions.squeeze(0).cpu().numpy().astype(np.int64),
        float(log_probability.item()),
        float(value.item()),
        action_mask.reshape(-1),
        policy_slot_mask.squeeze(0).cpu().numpy().astype(np.float32),
    )


def _potential(observation: Mapping[str, Any], learner_seat: int) -> float:
    """Bounded public/own-private liquidation proxy used only for shaping."""

    farms = observation.get("farms", [])
    if not isinstance(farms, list) or len(farms) != 2:
        return 0.0
    own = farms[learner_seat]
    private = observation.get("private", {}) or {}
    market_prices = ((observation.get("market", {}) or {}).get("prices", {}) or {})
    value = float(own.get("money", 0.0) or 0.0)
    for crop, count in (private.get("seeds", {}) or {}).items():
        value += SEED_COSTS.get(str(crop), 0) * max(0, int(count or 0))
    inventories = [private.get("shed", {}) or {}, *(private.get("inventories", []) or [])]
    for inventory in inventories:
        if not isinstance(inventory, Mapping):
            continue
        for item, count in inventory.items():
            unit_value = ANIMAL_COSTS.get(
                str(item), max(0, int(market_prices.get(str(item), 0) or 0))
            )
            value += unit_value * max(0, int(count or 0))
    opponent_money = float(farms[1 - learner_seat].get("money", 0.0) or 0.0)
    return math.tanh((value - opponent_money) / 10_000.0)


def _terminal_reward(
    score_difference: float,
    *,
    score_coefficient: float,
    score_scale: float,
) -> tuple[float, float]:
    """Return win outcome and a bounded dense terminal training reward."""

    if score_scale <= 0:
        raise ValueError("terminal_score_scale must be positive")
    outcome = 1.0 if score_difference > 0 else 0.0 if score_difference < 0 else 0.5
    win_reward = 2.0 * outcome - 1.0
    margin_reward = float(score_coefficient) * math.tanh(score_difference / score_scale)
    return outcome, win_reward + margin_reward


def _cpu_state_dict(model: Any) -> dict[str, Any]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _choose_snapshot(
    rng: random.Random,
    stats: list[dict[str, float]],
    *,
    power: float,
) -> int:
    weights = []
    for item in stats:
        games = int(item["games"])
        win_rate = float(item["wins"]) / games if games else 0.5
        weights.append(max(0.05, 1.0 - abs(2.0 * win_rate - 1.0)) ** power)
    return rng.choices(range(len(stats)), weights=weights, k=1)[0]


def _collect_episode(
    learner: Any,
    opponent: Any | None,
    opponent_policy: Callable[[dict[str, Any]], Mapping[str, Any]] | None,
    *,
    learner_seat: int,
    seed: int,
    gamma: float,
    gae_lambda: float,
    device: Any,
    tokenizer: ObservationTokenizer,
    codec: JointActionCodec,
    episode_steps: int,
    terminal_score_coefficient: float,
    terminal_score_scale: float,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    env = KaggricultureEnv(configuration={"episodeSteps": episode_steps})
    observations = env.reset(seed=seed)
    records: list[dict[str, Any]] = []
    final_rewards = {0: 0.0, 1: 0.0}

    learner.eval()
    if opponent is not None:
        opponent.eval()
    for _ in range(episode_steps):
        learner_encoded = _encoded(observations[learner_seat], tokenizer, codec)
        learner_indices, log_probability, value, learner_mask, policy_slot_mask = _sample(
            learner,
            learner_encoded,
            device=device,
            observation=observations[learner_seat],
            codec=codec,
        )
        learner_encoded["action_mask"] = learner_mask
        if opponent_policy is None:
            if opponent is None:
                raise ValueError("model or scripted opponent is required")
            opponent_encoded = _encoded(observations[1 - learner_seat], tokenizer, codec)
            opponent_indices, _, _, _, _ = _sample(
                opponent,
                opponent_encoded,
                device=device,
                observation=observations[1 - learner_seat],
                codec=codec,
            )
            opponent_action = codec.decode(
                observations[1 - learner_seat], opponent_indices
            )
        else:
            opponent_observation = observations[1 - learner_seat]
            farms = opponent_observation.get("farms", [])
            hand_count = len(farms[1 - learner_seat].get("hands", []) or [])
            opponent_action = validate_action(
                opponent_policy(deepcopy(opponent_observation)), hand_count=hand_count
            )
        actions = {
            learner_seat: codec.decode(observations[learner_seat], learner_indices),
            1 - learner_seat: opponent_action,
        }
        result = env.step(actions)
        done = result.terminated or result.truncated
        current_potential = _potential(observations[learner_seat], learner_seat)
        # A zero terminal potential makes the shaping telescope to a policy-
        # independent constant.  The old non-zero terminal potential changed
        # the objective into final cash difference and rewarded PASS-heavy
        # policies that avoided productive investment.
        next_potential = (
            0.0
            if done
            else _potential(result.observations[learner_seat], learner_seat)
        )
        reward = gamma * next_potential - current_potential
        final_rewards = result.rewards
        records.append(
            {
                **learner_encoded,
                "actions": learner_indices,
                "policy_slot_mask": policy_slot_mask,
                "old_log_probability": log_probability,
                "old_value": value,
                "reward": reward,
                "done": done,
            }
        )
        observations = result.observations
        if result.terminated or result.truncated:
            break

    score_difference = final_rewards[learner_seat] - final_rewards[1 - learner_seat]
    outcome, terminal_reward = _terminal_reward(
        score_difference,
        score_coefficient=terminal_score_coefficient,
        score_scale=terminal_score_scale,
    )
    if records:
        records[-1]["reward"] += terminal_reward

    # The official Kaggle environment retains its full replay history.  PPO
    # only needs the detached encoded records above, so release the environment
    # before the optimizer batch is assembled.
    env.close()

    advantage = 0.0
    next_value = 0.0
    for record in reversed(records):
        continuation = 0.0 if record["done"] else 1.0
        delta = (
            float(record["reward"])
            + gamma * next_value * continuation
            - float(record["old_value"])
        )
        advantage = delta + gamma * gae_lambda * continuation * advantage
        record["advantage"] = advantage
        record["return"] = advantage + float(record["old_value"])
        next_value = float(record["old_value"])
    return records, {
        "outcome": outcome,
        "score_difference": score_difference,
        "steps": float(len(records)),
    }


def _stack(records: list[dict[str, Any]]) -> dict[str, Any]:
    torch = _torch()
    result: dict[str, Any] = {}
    for key in ("tokens", "types", "attention", "action_mask", "actions", "policy_slot_mask"):
        result[key] = torch.from_numpy(np.stack([item[key] for item in records]))
    for key in ("old_log_probability", "old_value", "advantage", "return"):
        result[key] = torch.tensor([item[key] for item in records], dtype=torch.float32)
    advantages = result["advantage"]
    result["advantage"] = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)
    return result


def _update(
    model: Any,
    optimizer: Any,
    batch: Mapping[str, Any],
    *,
    device: Any,
    minibatch_size: int,
    epochs: int,
    clip_param: float,
    value_coeff: float,
    entropy_coeff: float,
    grad_clip: float,
    target_kl: float,
    reference_model: Any | None = None,
    reference_coeff: float = 0.0,
) -> dict[str, float]:
    """Apply a conservative PPO update with an optional frozen BC anchor.

    The anchor is intentionally evaluated on the exact prefix-conditioned masks
    stored in the rollout.  This prevents a large Transformer from drifting
    away from the legal, behaviour-cloned action distribution after a handful
    of noisy long-horizon games.
    """

    torch = _torch()
    if reference_coeff < 0:
        raise ValueError("reference_coeff must be non-negative")
    if reference_coeff and reference_model is None:
        raise ValueError("reference_coeff requires a frozen reference model")
    # Rollouts are sampled under ``learner.eval()``.  Transformer dropout must
    # remain disabled while recomputing their log probabilities, otherwise PPO
    # compares a deterministic behaviour policy with a different random policy
    # every minibatch and reports an artificial KL spike.
    model.eval()
    if reference_model is not None:
        reference_model.eval()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    sample_count = len(batch["advantage"])
    totals = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "kl": 0.0,
        "reference_kl": 0.0,
    }
    updates = 0
    stopped_early = False
    max_kl = 0.0
    for _ in range(epochs):
        indices = torch.randperm(sample_count)
        for start in range(0, sample_count, minibatch_size):
            selected = indices[start : start + minibatch_size]
            tensors = {key: value[selected].to(device, non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits, values = model(
                    tensors["tokens"],
                    tensors["types"],
                    tensors["attention"],
                    tensors["action_mask"],
                )
                distribution = torch.distributions.Categorical(logits=logits.float())
                policy_slot_mask = tensors["policy_slot_mask"].float()
                normalizer = policy_slot_mask.sum(dim=-1).clamp_min(1.0)
                log_probability = (
                    distribution.log_prob(tensors["actions"]) * policy_slot_mask
                ).sum(dim=-1) / normalizer
                ratio = (log_probability - tensors["old_log_probability"]).exp()
                unclipped = ratio * tensors["advantage"]
                clipped = ratio.clamp(1.0 - clip_param, 1.0 + clip_param) * tensors["advantage"]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = 0.5 * (values.float() - tensors["return"]).square().mean()
                entropy = (
                    (distribution.entropy() * policy_slot_mask).sum(dim=-1) / normalizer
                ).mean()
                reference_kl = torch.zeros((), dtype=torch.float32, device=device)
                if reference_model is not None:
                    with torch.no_grad():
                        reference_logits, _ = reference_model(
                            tensors["tokens"],
                            tensors["types"],
                            tensors["attention"],
                            tensors["action_mask"],
                        )
                    reference_distribution = torch.distributions.Categorical(
                        logits=reference_logits.float()
                    )
                    reference_kl = (
                        (
                            torch.distributions.kl_divergence(
                                distribution, reference_distribution
                            )
                            * policy_slot_mask
                        ).sum(dim=-1)
                        / normalizer
                    ).mean()
                loss = (
                    policy_loss
                    + value_coeff * value_loss
                    - entropy_coeff * entropy
                    + reference_coeff * reference_kl
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                # Measure KL after the optimizer step.  The old implementation
                # checked logits computed before that step, allowing a noisy
                # minibatch to overshoot the trust region by up to an epoch.
                post_logits, _ = model(
                    tensors["tokens"],
                    tensors["types"],
                    tensors["attention"],
                    tensors["action_mask"],
                )
                post_distribution = torch.distributions.Categorical(
                    logits=post_logits.float()
                )
                post_log_probability = (
                    post_distribution.log_prob(tensors["actions"]) * policy_slot_mask
                ).sum(dim=-1) / normalizer
                approximate_kl = (
                    tensors["old_log_probability"] - post_log_probability
                ).mean()
            totals["policy_loss"] += float(policy_loss.item())
            totals["value_loss"] += float(value_loss.item())
            totals["entropy"] += float(entropy.item())
            totals["kl"] += float(approximate_kl.item())
            totals["reference_kl"] += float(reference_kl.item())
            updates += 1
            max_kl = max(max_kl, float(approximate_kl.item()))
            if float(approximate_kl.item()) > target_kl:
                stopped_early = True
                break
        if stopped_early:
            break
    result = {key: value / max(1, updates) for key, value in totals.items()}
    result["kl_max"] = max_kl
    result["kl_early_stop"] = 1.0 if stopped_early else 0.0
    result["update_steps"] = float(updates)
    return result


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(value), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def run_native_self_play(
    config: Mapping[str, Any],
    *,
    iterations: int,
    output_dir: str | Path,
    resume: str | Path | None = None,
    bc_checkpoint: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Run checkpointed single-process PPO and return compact iteration metrics."""

    torch = _torch()
    training = dict(config.get("training", {}))
    native = dict(config.get("native", {}))
    self_play = dict(config.get("self_play", {}))
    model_config = ModelConfig.from_dict(dict(config.get("model", {})))
    device_name = str(native.get("device", "cuda")).lower()
    if device_name not in {"cpu", "cuda"}:
        raise ValueError("native.device must be cpu or cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("native PPO requested CUDA but CUDA is unavailable")
    device = torch.device(device_name)
    if device.type == "cpu":
        cpu_threads = int(native.get("cpu_threads", 4))
        if cpu_threads <= 0:
            raise ValueError("native.cpu_threads must be positive")
        torch.set_num_threads(cpu_threads)
    seed = int(native.get("seed", 20260825))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    model = build_actor_critic(model_config).to(device)
    opponent = build_actor_critic(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("lr", 3e-4)),
        weight_decay=float(native.get("weight_decay", 0.01)),
    )
    reference_model: Any | None = None
    reference_coeff = float(native.get("bc_anchor_coeff", 0.0))
    if reference_coeff < 0:
        raise ValueError("native.bc_anchor_coeff must be non-negative")
    snapshots = [_cpu_state_dict(model)]
    snapshot_stats: list[dict[str, float]] = [{"games": 0.0, "wins": 0.0}]
    promotion_outcomes: list[float] = []
    scripted_promotion_outcomes: list[float] = []
    start_iteration = 0
    if resume:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        if payload.get("format") != "farmer-native-ppo/v1":
            raise ValueError("unsupported native PPO checkpoint")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        # A recovery config is allowed to lower the step size; otherwise a
        # checkpoint silently restores the unstable optimizer learning rate.
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = float(training.get("lr", 3e-4))
        snapshots = payload.get("snapshots", snapshots)
        snapshot_stats = payload.get("snapshot_stats", snapshot_stats)
        promotion_outcomes = [float(value) for value in payload.get("promotion_outcomes", [])]
        scripted_promotion_outcomes = [
            float(value) for value in payload.get("scripted_promotion_outcomes", [])
        ]
        start_iteration = int(payload.get("iteration", 0))
    elif bc_checkpoint:
        payload = torch.load(bc_checkpoint, map_location="cpu", weights_only=False)
        if payload.get("format") != "farmer-rl-bc/v1":
            raise ValueError("unsupported behavior-cloning checkpoint")
        model.load_state_dict(payload["state_dict"], strict=True)
        snapshots = [_cpu_state_dict(model)]
        if reference_coeff:
            reference_model = deepcopy(model).eval()
            for parameter in reference_model.parameters():
                parameter.requires_grad_(False)
    elif reference_coeff:
        raise ValueError("native.bc_anchor_coeff requires --bc-checkpoint")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.jsonl"
    _write_json(output / "run_config.json", dict(config))

    gamma = float(training.get("gamma", 0.999))
    gae_lambda = float(training.get("gae_lambda", 0.95))
    target_steps = int(native.get("train_batch_steps", 2880))
    episode_steps = int(config.get("environment", {}).get("configuration", {}).get("episodeSteps", 720))
    terminal_score_coefficient = float(training.get("terminal_score_coeff", 0.0))
    terminal_score_scale = float(training.get("terminal_score_scale", 1000.0))
    if terminal_score_scale <= 0:
        raise ValueError("training.terminal_score_scale must be positive")
    tokenizer = ObservationTokenizer(max_tokens=model_config.max_tokens)
    codec = JointActionCodec(
        CandidateGenerator(capacity=model_config.candidate_capacity),
        max_hands=model_config.slots - 11,
        max_orders=10,
    )
    rng = random.Random(seed + start_iteration)
    history: list[dict[str, Any]] = []
    scripted_probability = float(native.get("scripted_opponent_probability", 0.5))
    if not 0.0 <= scripted_probability <= 1.0:
        raise ValueError("scripted_opponent_probability must be in [0, 1]")
    scripted_policy = None
    if scripted_probability > 0:
        from kaggle_environments.envs.kaggriculture.kaggriculture import starter_agent

        scripted_policy = starter_agent

    for iteration in range(start_iteration + 1, start_iteration + int(iterations) + 1):
        started = time.perf_counter()
        records: list[dict[str, Any]] = []
        outcomes: list[float] = []
        score_differences: list[float] = []
        scripted_outcomes: list[float] = []
        snapshot_outcomes: list[float] = []
        games = 0
        while len(records) < target_steps:
            use_scripted = scripted_policy is not None and rng.random() < scripted_probability
            snapshot_index: int | None = None
            if not use_scripted:
                snapshot_index = _choose_snapshot(
                    rng, snapshot_stats, power=float(self_play.get("pfsp_power", 1.0))
                )
                opponent.load_state_dict(snapshots[snapshot_index])
            learner_seat = games % 2
            episode_records, episode_metrics = _collect_episode(
                model,
                None if use_scripted else opponent,
                opponent_policy=scripted_policy if use_scripted else None,
                learner_seat=learner_seat,
                seed=seed + iteration * 10_000 + games,
                gamma=gamma,
                gae_lambda=gae_lambda,
                device=device,
                tokenizer=tokenizer,
                codec=codec,
                episode_steps=episode_steps,
                terminal_score_coefficient=terminal_score_coefficient,
                terminal_score_scale=terminal_score_scale,
            )
            records.extend(episode_records)
            outcome = float(episode_metrics["outcome"])
            outcomes.append(outcome)
            score_differences.append(float(episode_metrics["score_difference"]))
            if use_scripted:
                scripted_outcomes.append(outcome)
                scripted_promotion_outcomes.append(outcome)
            else:
                assert snapshot_index is not None
                snapshot_outcomes.append(outcome)
                promotion_outcomes.append(outcome)
                snapshot_stats[snapshot_index]["games"] += 1.0
                snapshot_stats[snapshot_index]["wins"] += outcome
            games += 1

        update_metrics = _update(
            model,
            optimizer,
            _stack(records),
            device=device,
            minibatch_size=int(native.get("minibatch_size", 32)),
            epochs=int(native.get("update_epochs", 4)),
            clip_param=float(training.get("clip_param", 0.2)),
            value_coeff=float(native.get("value_coeff", 0.5)),
            entropy_coeff=float(native.get("entropy_coeff", 0.002)),
            grad_clip=float(native.get("grad_clip", 1.0)),
            target_kl=float(native.get("target_kl", 0.03)),
            reference_model=reference_model,
            reference_coeff=reference_coeff,
        )
        promotion_interval = int(self_play.get("promotion_interval", 5))
        pool_size = int(self_play.get("checkpoint_slots", 4))
        promotion_win_rate = float(np.mean(promotion_outcomes)) if promotion_outcomes else 0.0
        promotion_due = iteration % promotion_interval == 0
        promotion_min_games = int(self_play.get("promotion_min_games", 8))
        promotion_threshold = float(self_play.get("promotion_win_rate", 0.55))
        promotion_scripted_min_games = int(
            self_play.get("promotion_scripted_min_games", 0)
        )
        promotion_scripted_threshold = float(
            self_play.get("promotion_scripted_win_rate", 0.0)
        )
        promotion_scripted_win_rate = (
            float(np.mean(scripted_promotion_outcomes))
            if scripted_promotion_outcomes
            else 0.0
        )
        scripted_gate_passed = (
            len(scripted_promotion_outcomes) >= promotion_scripted_min_games
            and promotion_scripted_win_rate >= promotion_scripted_threshold
        )
        promoted = (
            promotion_due
            and len(promotion_outcomes) >= promotion_min_games
            and promotion_win_rate >= promotion_threshold
            and scripted_gate_passed
        )
        if promoted:
            snapshots.append(_cpu_state_dict(model))
            snapshot_stats.append({"games": 0.0, "wins": 0.0})
            promotion_outcomes.clear()
            scripted_promotion_outcomes.clear()
            if len(snapshots) > pool_size:
                snapshots.pop(0)
                snapshot_stats.pop(0)
        elif promotion_due:
            promotion_outcomes = promotion_outcomes[-max(promotion_min_games * 4, 32) :]
            scripted_promotion_outcomes = scripted_promotion_outcomes[
                -max(promotion_scripted_min_games * 4, 32) :
            ]

        elapsed = time.perf_counter() - started
        metrics = {
            "iteration": iteration,
            "seconds": elapsed,
            "steps": len(records),
            "steps_per_second": len(records) / max(elapsed, 1e-6),
            "games": games,
            "learner_win_rate": float(np.mean(outcomes)),
            "scripted_games": len(scripted_outcomes),
            "scripted_win_rate": float(np.mean(scripted_outcomes)) if scripted_outcomes else None,
            "snapshot_games": len(snapshot_outcomes),
            "snapshot_win_rate": float(np.mean(snapshot_outcomes)) if snapshot_outcomes else None,
            "promotion_window_games": len(promotion_outcomes),
            "promotion_window_win_rate": promotion_win_rate,
            "promotion_scripted_games": len(scripted_promotion_outcomes),
            "promotion_scripted_win_rate": promotion_scripted_win_rate,
            "promotion_scripted_gate_passed": scripted_gate_passed,
            "promoted": promoted,
            "mean_score_difference": float(np.mean(score_differences)),
            "pool_size": len(snapshots),
            "device": device.type,
            "cuda_peak_gib": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0,
            "kl_early_stop": bool(update_metrics.pop("kl_early_stop")),
            **update_metrics,
        }
        history.append(metrics)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        print(json.dumps(metrics, ensure_ascii=False), flush=True)

        checkpoint_interval = int(native.get("checkpoint_interval", 5))
        if iteration % checkpoint_interval == 0 or iteration == start_iteration + int(iterations):
            checkpoint_path = checkpoints / f"iteration_{iteration:06d}.pt"
            torch.save(
                {
                    "format": "farmer-native-ppo/v1",
                    "iteration": iteration,
                    "model_config": model_config.__dict__,
                    "model": _cpu_state_dict(model),
                    "optimizer": optimizer.state_dict(),
                    "snapshots": snapshots,
                    "snapshot_stats": snapshot_stats,
                    "promotion_outcomes": promotion_outcomes,
                    "scripted_promotion_outcomes": scripted_promotion_outcomes,
                },
                checkpoint_path,
            )
            _write_json(
                output / "latest.json",
                {"iteration": iteration, "checkpoint": str(checkpoint_path.resolve())},
            )
            keep = max(1, int(native.get("keep_checkpoints", 8)))
            checkpoint_files = sorted(checkpoints.glob("iteration_*.pt"))
            for obsolete in checkpoint_files[:-keep]:
                obsolete.unlink()
    return history
