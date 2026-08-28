"""Residual Transformer Actor-Critic used by BC and RLlib PPO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import OptionalDependencyError
from .tokenizer import FEATURE_DIM, TOKEN_TYPES


def _torch() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise OptionalDependencyError(
            "Transformer training requires PyTorch. Install a build appropriate for your platform."
        ) from exc
    return torch, nn


@dataclass(frozen=True)
class ModelConfig:
    feature_dim: int = FEATURE_DIM
    max_tokens: int = 320
    slots: int = 27
    candidate_capacity: int = 64
    d_model: int = 256
    nhead: int = 8
    layers: int = 8
    dim_feedforward: int = 1024
    dropout: float = 0.1
    # PPG highlights destructive interference when policy and value losses
    # share a representation.  Keep one large residual Transformer, but allow
    # the critic gradient reaching that shared encoder to be attenuated.  The
    # critic head itself still receives its full gradient.
    critic_encoder_gradient_scale: float = 1.0

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModelConfig":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: item for key, item in value.items() if key in allowed})


def build_actor_critic(config: ModelConfig) -> Any:
    """Build lazily so importing core schemas does not require PyTorch."""

    torch, nn = _torch()

    class ResidualTransformerActorCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            if not 0.0 <= config.critic_encoder_gradient_scale <= 1.0:
                raise ValueError("critic_encoder_gradient_scale must be in [0, 1]")
            self.input_projection = nn.Linear(config.feature_dim, config.d_model)
            self.type_embedding = nn.Embedding(len(TOKEN_TYPES), config.d_model)
            self.position_embedding = nn.Parameter(torch.zeros(1, config.max_tokens, config.d_model))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.layers)
            self.final_norm = nn.LayerNorm(config.d_model)
            self.slot_embedding = nn.Parameter(torch.empty(1, config.slots, config.d_model))
            nn.init.normal_(self.slot_embedding, std=0.02)
            self.actor = nn.Sequential(
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, config.candidate_capacity),
            )
            self.critic = nn.Sequential(
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, 1),
            )

        def forward(self, values: Any, token_type_ids: Any, attention_mask: Any, action_mask: Any | None = None) -> tuple[Any, Any]:
            if values.ndim != 3:
                raise ValueError("values must have shape [batch, tokens, features]")
            batch, tokens, _ = values.shape
            if tokens > self.config.max_tokens:
                raise ValueError("token sequence exceeds configured max_tokens")
            hidden = self.input_projection(values)
            hidden = hidden + self.type_embedding(token_type_ids.long())
            hidden = hidden + self.position_embedding[:, :tokens]
            padding_mask = attention_mask <= 0
            hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
            hidden = self.final_norm(hidden)
            weights = attention_mask.to(hidden.dtype).unsqueeze(-1)
            pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            slot_hidden = pooled.unsqueeze(1) + self.slot_embedding
            logits = self.actor(slot_hidden)
            if action_mask is not None:
                if action_mask.ndim == 2:
                    action_mask = action_mask.reshape(batch, self.config.slots, self.config.candidate_capacity)
                logits = logits.masked_fill(action_mask <= 0, torch.finfo(logits.dtype).min)
            # Forward values are unchanged, while the gradient from the value
            # loss into the shared Transformer is scaled independently from
            # the critic head gradient.  This is checkpoint-compatible because
            # it introduces no parameters.
            scale = self.config.critic_encoder_gradient_scale
            critic_pooled = pooled.detach() + scale * (pooled - pooled.detach())
            value = self.critic(critic_pooled).squeeze(-1)
            return logits, value

    return ResidualTransformerActorCritic()
