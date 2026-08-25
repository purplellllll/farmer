"""Behaviour-cloning warm start from versioned local trajectory JSONL."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .actions import CandidateGenerator, JointActionCodec
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
) -> dict[str, Any]:
    try:
        import torch
        from torch.nn import functional as functional
    except ImportError as exc:  # pragma: no cover
        raise OptionalDependencyError("BC requires PyTorch") from exc

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
    examples: list[tuple[Any, ...]] = []
    skipped = 0
    for transition in iter_transitions(input_paths):
        try:
            labels = codec.encode(transition.observation, transition.action)
        except InvalidActionError:
            # Conservative candidates intentionally reject unsupported expert
            # actions. Count rather than silently relabel them as PASS.
            skipped += 1
            continue
        batch = tokenizer.tokenize(transition.observation)
        farms = transition.observation.get("farms", [])
        own_farm = farms[transition.acting_seat]
        active_hands = len(own_farm.get("hands", []) or [])
        market_count = min(len(transition.action.get("market", []) or []), 10)
        decision_weights = np.zeros(model_config.slots, dtype=np.float32)
        decision_weights[: 1 + active_hands] = 1.0
        market_start = 1 + max_hands
        # Supervise every emitted order and one NO_ORDER stop token.  Do not let
        # the remaining padded queue slots overwhelm rare expert decisions.
        decision_weights[market_start : market_start + 10] = 0.05
        decision_weights[market_start : market_start + min(10, market_count + 1)] = 0.5
        examples.append(
            (batch, labels, codec.mask(transition.observation), decision_weights)
        )
    if not examples:
        raise ValueError("no encodable BC examples; inspect source license, schema, and candidate coverage")

    counts = np.zeros((model_config.slots, model_config.candidate_capacity), dtype=np.float64)
    for _, labels, _, decision_weights in examples:
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
        order = torch.randperm(len(examples)).tolist()
        for start in range(0, len(order), int(batch_size)):
            selected = [examples[index] for index in order[start : start + int(batch_size)]]
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

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "farmer-rl-bc/v1",
            "model_config": asdict(model_config),
            "state_dict": model.state_dict(),
            "examples": len(examples),
            "skipped": skipped,
        },
        output,
    )
    return {
        "examples": len(examples),
        "skipped": skipped,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "final_loss": final_loss,
        "output": str(output),
    }
