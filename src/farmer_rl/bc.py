"""Behaviour-cloning warm start from versioned local trajectory JSONL."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

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
        examples.append((batch, labels, codec.mask(transition.observation)))
    if not examples:
        raise ValueError("no encodable BC examples; inspect source license, schema, and candidate coverage")

    model.train()
    final_loss = 0.0
    for _ in range(int(epochs)):
        for batch, labels, mask in examples:
            values = torch.tensor(batch.values, dtype=torch.float32, device=device).unsqueeze(0)
            token_types = torch.tensor(batch.token_type_ids, dtype=torch.long, device=device).unsqueeze(0)
            attention = torch.tensor(batch.attention_mask, dtype=torch.float32, device=device).unsqueeze(0)
            action_mask = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)
            targets = torch.tensor(labels, dtype=torch.long, device=device).unsqueeze(0)
            logits, _ = model(values, token_types, attention, action_mask)
            loss = functional.cross_entropy(
                logits.reshape(-1, model_config.candidate_capacity), targets.reshape(-1)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
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
    return {"examples": len(examples), "skipped": skipped, "final_loss": final_loss, "output": str(output)}
