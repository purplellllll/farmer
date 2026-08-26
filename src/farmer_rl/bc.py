"""Behaviour-cloning warm start from versioned local trajectory JSONL."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .actions import CandidateGenerator, JointActionCodec
from .environment import validate_action
from .errors import InvalidActionError, OptionalDependencyError
from .model import ModelConfig, build_actor_critic
from .tokenizer import ObservationTokenizer
from .trajectory import Transition


def iter_transitions(paths: Iterable[str | Path]) -> Iterable[Transition]:
    for path_value in paths:
        path = Path(path_value)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    yield Transition.from_dict(json.loads(line))
                except Exception as exc:
                    raise ValueError(f"invalid trajectory record {path}:{line_number}: {exc}") from exc


def iter_bc_records(paths: Iterable[str | Path]) -> Iterable[dict[str, Any]]:
    """Yield the minimal observation/action schema accepted by BC.

    The original collector stores full :class:`Transition` objects.  The
    curriculum-v2 generator intentionally stores observation/action pairs
    because next-state and reward are irrelevant to supervised warm-starting.
    Both are validated here without manufacturing fictitious RL transitions.
    """

    for path_value in paths:
        path = Path(path_value)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    if "next_observation" in value:
                        transition = Transition.from_dict(value)
                        yield {
                            "episode_id": transition.episode_id,
                            "seed": transition.seed,
                            "acting_seat": transition.acting_seat,
                            "step": transition.step,
                            "observation": transition.observation,
                            "action": transition.action,
                        }
                        continue
                    observation = value["observation"]
                    action = validate_action(value["action"])
                    acting_seat = int(value["acting_seat"])
                    if acting_seat not in (0, 1) or int(observation.get("player", -1)) != acting_seat:
                        raise ValueError("observation.player must equal acting_seat")
                    episode_id = str(value["episode_id"])
                    if not episode_id:
                        raise ValueError("episode_id is required")
                    yield {
                        "episode_id": episode_id,
                        "seed": value.get("seed"),
                        "acting_seat": acting_seat,
                        "step": int(value.get("step", 0)),
                        "observation": observation,
                        "action": action,
                    }
                except Exception as exc:
                    raise ValueError(f"invalid BC record {path}:{line_number}: {exc}") from exc


def _validation_group(record: Mapping[str, Any], *, fraction: float, split_seed: int) -> bool:
    """Assign a whole episode+seed group to validation deterministically."""

    key = f"{record['episode_id']}|{record.get('seed')}|{split_seed}".encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / float(2**64)
    return bucket < fraction


def train_bc(
    *,
    input_paths: list[str | Path],
    output_path: str | Path,
    model_config: ModelConfig,
    epochs: int = 3,
    learning_rate: float = 3e-4,
    device: str = "cpu",
    batch_size: int = 32,
    initial_checkpoint: str | Path | None = None,
    validation_fraction: float = 0.2,
    split_seed: int = 20260826,
    max_train_examples: int | None = None,
    max_validation_examples: int | None = None,
    max_examples_per_group: int | None = None,
    record_sample_modulus: int = 1,
    torch_threads: int | None = None,
) -> dict[str, Any]:
    try:
        import torch
        from torch.nn import functional as functional
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError("BC requires PyTorch") from exc

    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    if int(record_sample_modulus) <= 0:
        raise ValueError("record_sample_modulus must be positive")
    if torch_threads is not None:
        if int(torch_threads) <= 0:
            raise ValueError("torch_threads must be positive")
        torch.set_num_threads(int(torch_threads))
    torch.manual_seed(int(split_seed))
    tokenizer = ObservationTokenizer(max_tokens=model_config.max_tokens)
    max_hands = model_config.slots - 11
    codec = JointActionCodec(
        CandidateGenerator(capacity=model_config.candidate_capacity), max_hands=max_hands, max_orders=10
    )
    model = build_actor_critic(model_config).to(device)
    if initial_checkpoint:
        initial = torch.load(initial_checkpoint, map_location="cpu", weights_only=False)
        if initial.get("format") != "farmer-rl-bc/v1":
            raise ValueError("initial checkpoint is not farmer-rl-bc/v1")
        model.load_state_dict(initial["state_dict"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_examples: list[tuple[Any, ...]] = []
    validation_examples: list[tuple[Any, ...]] = []
    train_groups: set[tuple[str, Any]] = set()
    validation_groups: set[tuple[str, Any]] = set()
    skipped = 0
    skipped_train = 0
    skipped_validation = 0
    group_example_counts: dict[tuple[str, Any], int] = {}
    for record in iter_bc_records(input_paths):
        is_validation = _validation_group(
            record, fraction=float(validation_fraction), split_seed=int(split_seed)
        )
        target_examples = validation_examples if is_validation else train_examples
        target_limit = max_validation_examples if is_validation else max_train_examples
        group = (str(record["episode_id"]), record.get("seed"))
        sample_key = (
            f"{group[0]}|{group[1]}|{record['acting_seat']}|{record.get('step', 0)}|{split_seed}"
        ).encode("utf-8")
        if int.from_bytes(hashlib.sha256(sample_key).digest()[:8], "big") % int(record_sample_modulus):
            continue
        if (
            max_examples_per_group is not None
            and group_example_counts.get(group, 0) >= int(max_examples_per_group)
        ):
            continue
        if target_limit is not None and len(target_examples) >= int(target_limit):
            train_done = (
                max_train_examples is not None
                and len(train_examples) >= int(max_train_examples)
            )
            validation_done = validation_fraction <= 0 or (
                max_validation_examples is not None
                and len(validation_examples) >= int(max_validation_examples)
            )
            if train_done and validation_done:
                break
            continue
        observation = record["observation"]
        action = record["action"]
        try:
            labels = codec.encode(observation, action)
        except InvalidActionError:
            # Conservative candidates intentionally reject unsupported expert
            # actions. Count rather than silently relabel them as PASS.
            skipped += 1
            if is_validation:
                skipped_validation += 1
            else:
                skipped_train += 1
            continue
        tokenized = tokenizer.tokenize(observation)
        farms = observation.get("farms", [])
        own_farm = farms[int(record["acting_seat"])]
        active_hands = len(own_farm.get("hands", []) or [])
        market_count = min(len(action.get("market", []) or []), 10)
        decision_weights = np.zeros(model_config.slots, dtype=np.float32)
        decision_weights[: 1 + active_hands] = 1.0
        market_start = 1 + max_hands
        # Supervise every emitted order and one NO_ORDER stop token.  Do not let
        # the remaining padded queue slots overwhelm rare expert decisions.
        decision_weights[market_start : market_start + 10] = 0.05
        decision_weights[market_start : market_start + min(10, market_count + 1)] = 0.5
        target_examples.append(
            (tokenized, labels, codec.mask(observation), decision_weights)
        )
        group_example_counts[group] = group_example_counts.get(group, 0) + 1
        (validation_groups if is_validation else train_groups).add(group)
    if not train_examples:
        raise ValueError("no encodable BC examples; inspect source license, schema, and candidate coverage")

    counts = np.zeros((model_config.slots, model_config.candidate_capacity), dtype=np.float64)
    for _, labels, _, decision_weights in train_examples:
        for slot, label in enumerate(labels):
            if decision_weights[slot] > 0:
                counts[slot, int(label)] += float(decision_weights[slot])
    max_counts = counts.max(axis=1, keepdims=True).clip(min=1)
    balanced = (max_counts / counts.clip(min=1)).clip(max=50.0).astype(np.float32)

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda"))
    final_loss = 0.0
    for _ in range(int(epochs)):
        order = torch.randperm(len(train_examples)).tolist()
        for start in range(0, len(order), int(batch_size)):
            selected = [train_examples[index] for index in order[start : start + int(batch_size)]]
            values = torch.tensor(
                [item[0].values for item in selected], dtype=torch.float32, device=device
            )
            token_types = torch.tensor(
                [item[0].token_type_ids for item in selected], dtype=torch.long, device=device
            )
            attention = torch.tensor(
                [item[0].attention_mask for item in selected], dtype=torch.bool, device=device
            )
            action_mask = torch.tensor(
                [item[2] for item in selected], dtype=torch.bool, device=device
            )
            targets = torch.tensor(
                [item[1] for item in selected], dtype=torch.long, device=device
            )
            decision_weights = torch.tensor(
                np.stack([item[3] for item in selected]), dtype=torch.float32, device=device
            )
            class_weights = torch.tensor(balanced, dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda" if str(device).startswith("cuda") else "cpu",
                dtype=torch.float16,
                enabled=str(device).startswith("cuda"),
            ):
                logits, _ = model(values, token_types, attention, action_mask)
                per_slot = functional.cross_entropy(
                    logits.reshape(-1, model_config.candidate_capacity),
                    targets.reshape(-1),
                    reduction="none",
                ).reshape(len(selected), model_config.slots)
                selected_class_weights = class_weights.unsqueeze(0).expand(
                    len(selected), -1, -1
                ).gather(2, targets.unsqueeze(-1)).squeeze(-1)
                effective_weights = decision_weights * selected_class_weights
                loss = (per_slot * effective_weights).sum() / effective_weights.sum().clamp_min(1.0)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            final_loss = float(loss.detach().cpu())

    validation_loss: float | None = None
    validation_slot_accuracy: float | None = None
    validation_joint_accuracy: float | None = None
    if validation_examples:
        model.eval()
        total_loss = 0.0
        total_weight = 0.0
        correct_weight = 0.0
        joint_correct = 0
        with torch.inference_mode():
            for start in range(0, len(validation_examples), int(batch_size)):
                selected = validation_examples[start : start + int(batch_size)]
                values = torch.tensor([item[0].values for item in selected], dtype=torch.float32, device=device)
                token_types = torch.tensor([item[0].token_type_ids for item in selected], dtype=torch.long, device=device)
                attention = torch.tensor([item[0].attention_mask for item in selected], dtype=torch.bool, device=device)
                action_mask = torch.tensor([item[2] for item in selected], dtype=torch.bool, device=device)
                targets = torch.tensor([item[1] for item in selected], dtype=torch.long, device=device)
                weights = torch.tensor(np.stack([item[3] for item in selected]), dtype=torch.float32, device=device)
                logits, _ = model(values, token_types, attention, action_mask)
                per_slot = functional.cross_entropy(
                    logits.reshape(-1, model_config.candidate_capacity),
                    targets.reshape(-1),
                    reduction="none",
                ).reshape(len(selected), model_config.slots)
                predictions = logits.argmax(dim=-1)
                total_loss += float((per_slot * weights).sum().item())
                total_weight += float(weights.sum().item())
                correct_weight += float(((predictions == targets) * weights).sum().item())
                supervised = weights > 0
                joint_correct += int((((predictions == targets) | ~supervised).all(dim=1)).sum().item())
        validation_loss = total_loss / max(1.0, total_weight)
        validation_slot_accuracy = correct_weight / max(1.0, total_weight)
        validation_joint_accuracy = joint_correct / len(validation_examples)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "farmer-rl-bc/v1",
            "model_config": asdict(model_config),
            "state_dict": model.state_dict(),
            "examples": len(train_examples),
            "skipped": skipped,
            "validation": {
                "examples": len(validation_examples),
                "loss": validation_loss,
                "slot_accuracy": validation_slot_accuracy,
                "joint_accuracy": validation_joint_accuracy,
            },
            "data_split": {
                "unit": "episode_id+seed",
                "validation_fraction": float(validation_fraction),
                "split_seed": int(split_seed),
                "train_groups": len(train_groups),
                "validation_groups": len(validation_groups),
                "group_overlap": len(train_groups & validation_groups),
                "max_examples_per_group": max_examples_per_group,
                "record_sample_modulus": int(record_sample_modulus),
            },
        },
        output,
    )
    return {
        "examples": len(train_examples),
        "skipped": skipped,
        "skipped_train": skipped_train,
        "skipped_validation": skipped_validation,
        "validation_examples": len(validation_examples),
        "validation_loss": validation_loss,
        "validation_slot_accuracy": validation_slot_accuracy,
        "validation_joint_accuracy": validation_joint_accuracy,
        "split_unit": "episode_id+seed",
        "train_groups": len(train_groups),
        "validation_groups": len(validation_groups),
        "group_overlap": len(train_groups & validation_groups),
        "split_seed": int(split_seed),
        "max_examples_per_group": max_examples_per_group,
        "record_sample_modulus": int(record_sample_modulus),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "final_loss": final_loss,
        "output": str(output),
    }
