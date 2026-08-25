"""Lazy RLlib PPO adapter and checkpoint-pool self-play driver.

RLlib changes APIs frequently.  The integration uses the stable old-stack
``TorchModelV2`` seam available in Ray 2.x and fails with a versioned message
when Ray/Gymnasium are absent.  Core collection and tests do not depend on it.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from typing import Any, Mapping

from .actions import CandidateGenerator, JointActionCodec
from .environment import KaggricultureEnv
from .errors import OptionalDependencyError
from .model import ModelConfig, build_actor_critic
from .tokenizer import FEATURE_DIM, ObservationTokenizer


def _dependencies() -> dict[str, Any]:
    try:
        import gymnasium as gym
        import numpy as np
        import ray
        import torch
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.rllib.env.multi_agent_env import MultiAgentEnv
        from ray.rllib.models import ModelCatalog
        from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
        from ray.tune.registry import register_env
    except ImportError as exc:  # pragma: no cover - optional path
        raise OptionalDependencyError(
            "RLlib PPO requires optional dependencies. Install a compatible Ray 2.x stack: "
            "pip install 'ray[rllib]>=2.40,<3' gymnasium torch "
            "and kaggle-environments==1.32.7"
        ) from exc
    return locals()


def _spaces(gym: Any, np: Any, model_config: ModelConfig) -> tuple[Any, Any]:
    observation_space = gym.spaces.Dict(
        {
            "tokens": gym.spaces.Box(
                low=-10.0,
                high=10.0,
                shape=(model_config.max_tokens, FEATURE_DIM),
                dtype=np.float32,
            ),
            "token_type_ids": gym.spaces.Box(
                low=0, high=32, shape=(model_config.max_tokens,), dtype=np.int64
            ),
            "attention_mask": gym.spaces.Box(
                low=0, high=1, shape=(model_config.max_tokens,), dtype=np.int8
            ),
            "action_mask": gym.spaces.Box(
                low=0,
                high=1,
                shape=(model_config.slots * model_config.candidate_capacity,),
                dtype=np.int8,
            ),
        }
    )
    action_space = gym.spaces.MultiDiscrete(
        [model_config.candidate_capacity] * model_config.slots
    )
    return observation_space, action_space


def make_rllib_env_class(model_config: ModelConfig) -> Any:
    deps = _dependencies()
    gym, np, MultiAgentEnv = deps["gym"], deps["np"], deps["MultiAgentEnv"]
    observation_space, action_space = _spaces(gym, np, model_config)

    class KaggricultureMultiAgentEnv(MultiAgentEnv):
        def __init__(self, env_config: Mapping[str, Any] | None = None) -> None:
            super().__init__()
            env_config = dict(env_config or {})
            self.core = KaggricultureEnv(
                configuration=env_config.get("configuration"),
                debug=bool(env_config.get("debug", False)),
            )
            self.tokenizer = ObservationTokenizer(max_tokens=model_config.max_tokens)
            self.codec = JointActionCodec(
                CandidateGenerator(capacity=model_config.candidate_capacity),
                max_hands=model_config.slots - 11,
                max_orders=10,
            )
            self.observation_space = observation_space
            self.action_space = action_space
            self.observation_spaces = {f"seat_{seat}": observation_space for seat in (0, 1)}
            self.action_spaces = {f"seat_{seat}": action_space for seat in (0, 1)}
            self._observations: dict[int, dict[str, Any]] = {}

        def _encode_observation(self, seat: int) -> dict[str, Any]:
            raw = self._observations[seat]
            batch = self.tokenizer.tokenize(raw)
            return {
                "tokens": np.asarray(batch.values, dtype=np.float32),
                "token_type_ids": np.asarray(batch.token_type_ids, dtype=np.int64),
                "attention_mask": np.asarray(batch.attention_mask, dtype=np.int8),
                "action_mask": np.asarray(self.codec.mask(raw), dtype=np.int8),
            }

        def reset(self, *, seed: int | None = None, options: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
            del options
            self._observations = self.core.reset(seed=seed)
            observations = {f"seat_{seat}": self._encode_observation(seat) for seat in (0, 1)}
            return observations, {agent_id: {} for agent_id in observations}

        def step(self, actions: Mapping[str, Any]) -> tuple[Any, ...]:
            raw_actions = {
                seat: self.codec.decode(self._observations[seat], actions[f"seat_{seat}"])
                for seat in (0, 1)
            }
            result = self.core.step(raw_actions)
            self._observations = result.observations
            observations = {f"seat_{seat}": self._encode_observation(seat) for seat in (0, 1)}
            rewards = {f"seat_{seat}": result.rewards[seat] for seat in (0, 1)}
            terminated = {f"seat_{seat}": result.terminated for seat in (0, 1)}
            terminated["__all__"] = result.terminated
            truncated = {f"seat_{seat}": result.truncated for seat in (0, 1)}
            truncated["__all__"] = result.truncated
            infos = {
                f"seat_{seat}": {"status": result.statuses[seat], **result.info}
                for seat in (0, 1)
            }
            return observations, rewards, terminated, truncated, infos

    return KaggricultureMultiAgentEnv


def make_rllib_model_class(model_config: ModelConfig) -> Any:
    deps = _dependencies()
    torch, TorchModelV2 = deps["torch"], deps["TorchModelV2"]
    nn = torch.nn

    class MaskedTransformerModel(TorchModelV2, nn.Module):
        def __init__(self, obs_space: Any, action_space: Any, num_outputs: int, model_config_dict: Any, name: str) -> None:
            TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config_dict, name)
            nn.Module.__init__(self)
            expected = model_config.slots * model_config.candidate_capacity
            if num_outputs != expected:
                raise ValueError(f"RLlib requested {num_outputs} logits; expected {expected}")
            self.core = build_actor_critic(model_config)
            self._value = None

        def forward(self, input_dict: Mapping[str, Any], state: list[Any], seq_lens: Any) -> tuple[Any, list[Any]]:
            del seq_lens
            obs = input_dict["obs"]
            logits, self._value = self.core(
                obs["tokens"],
                obs["token_type_ids"],
                obs["attention_mask"],
                obs["action_mask"],
            )
            return logits.reshape(logits.shape[0], -1), state

        def value_function(self) -> Any:
            if self._value is None:
                raise RuntimeError("forward must run before value_function")
            return self._value

    return MaskedTransformerModel


def build_ppo(config: Mapping[str, Any]) -> Any:
    """Build (but do not start training) a two-seat PPO algorithm."""

    deps = _dependencies()
    PPOConfig = deps["PPOConfig"]
    ModelCatalog = deps["ModelCatalog"]
    register_env = deps["register_env"]
    model_config = ModelConfig.from_dict(dict(config.get("model", {})))
    env_name = "farmer_rl_kaggriculture_v1"
    model_name = "farmer_rl_residual_transformer_v1"
    env_class = make_rllib_env_class(model_config)
    model_class = make_rllib_model_class(model_config)
    register_env(env_name, lambda env_config: env_class(env_config))
    ModelCatalog.register_custom_model(model_name, model_class)

    pool_size = int(config.get("self_play", {}).get("checkpoint_slots", 4))
    opponent_ids = tuple(f"opponent_{index}" for index in range(pool_size))
    policies = {"learner", *opponent_ids}

    def policy_mapping(agent_id: str, episode: Any, **kwargs: Any) -> str:
        del kwargs
        raw_episode_id = str(getattr(episode, "episode_id", "0"))
        try:
            episode_number = int(raw_episode_id)
        except ValueError:
            episode_number = int.from_bytes(sha256(raw_episode_id.encode("utf-8")).digest()[:8], "big")
        learner_seat = episode_number % 2
        if agent_id == f"seat_{learner_seat}":
            return "learner"
        return opponent_ids[episode_number % len(opponent_ids)]

    training = config.get("training", {})
    base = PPOConfig()
    if hasattr(base, "api_stack"):
        base = base.api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
    algorithm = (
        base
        .environment(env=env_name, env_config=dict(config.get("environment", {})))
        .framework("torch")
        .training(
            lr=float(training.get("lr", 3e-4)),
            gamma=float(training.get("gamma", 0.999)),
            lambda_=float(training.get("gae_lambda", 0.95)),
            clip_param=float(training.get("clip_param", 0.2)),
            train_batch_size=int(training.get("train_batch_size", 14400)),
            model={"custom_model": model_name},
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping,
            policies_to_train=["learner"],
        )
    )
    return algorithm.build()


def load_bc_checkpoint(algorithm: Any, path: str) -> None:
    deps = _dependencies()
    torch = deps["torch"]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "farmer-rl-bc/v1":
        raise ValueError("not a farmer-rl BC checkpoint")
    learner = algorithm.get_policy("learner")
    learner.model.core.load_state_dict(payload["state_dict"], strict=True)


def run_self_play(
    config: Mapping[str, Any], *, iterations: int, bc_checkpoint: str | None = None
) -> list[dict[str, Any]]:
    """Train PPO and periodically copy the learner into a frozen pool slot."""

    algorithm = build_ppo(config)
    if bc_checkpoint:
        load_bc_checkpoint(algorithm, bc_checkpoint)
    promotion_interval = int(config.get("self_play", {}).get("promotion_interval", 5))
    pool_size = int(config.get("self_play", {}).get("checkpoint_slots", 4))
    history: list[dict[str, Any]] = []
    try:
        for iteration in range(1, int(iterations) + 1):
            result = algorithm.train()
            history.append(deepcopy(result))
            if iteration % promotion_interval == 0:
                slot = ((iteration // promotion_interval) - 1) % pool_size
                algorithm.get_policy(f"opponent_{slot}").set_weights(
                    algorithm.get_policy("learner").get_weights()
                )
        return history
    finally:
        algorithm.stop()
