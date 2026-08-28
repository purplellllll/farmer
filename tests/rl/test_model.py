from __future__ import annotations

import importlib.util
import unittest

from farmer_rl.model import ModelConfig, build_actor_critic
from farmer_rl.tokenizer import FEATURE_DIM


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is optional")
class ModelTests(unittest.TestCase):
    def test_actor_critic_shapes_and_mask(self):
        import torch

        config = ModelConfig(
            feature_dim=FEATURE_DIM,
            max_tokens=8,
            slots=3,
            candidate_capacity=5,
            d_model=16,
            nhead=4,
            layers=1,
            dim_feedforward=32,
            dropout=0.0,
        )
        model = build_actor_critic(config)
        values = torch.zeros(2, 8, FEATURE_DIM)
        token_types = torch.zeros(2, 8, dtype=torch.long)
        attention = torch.ones(2, 8)
        action_mask = torch.ones(2, 15)
        action_mask[:, 4] = 0
        logits, value = model(values, token_types, attention, action_mask)
        self.assertEqual(tuple(logits.shape), (2, 3, 5))
        self.assertEqual(tuple(value.shape), (2,))
        self.assertTrue(torch.all(logits[:, 0, 4] < -1e20))

    def test_critic_encoder_gradient_can_be_attenuated(self):
        import torch

        def encoder_grad(scale: float) -> float:
            torch.manual_seed(7)
            config = ModelConfig(
                feature_dim=FEATURE_DIM,
                max_tokens=4,
                slots=1,
                candidate_capacity=2,
                d_model=8,
                nhead=2,
                layers=1,
                dim_feedforward=16,
                dropout=0.0,
                critic_encoder_gradient_scale=scale,
            )
            model = build_actor_critic(config)
            values = torch.randn(2, 4, FEATURE_DIM)
            types = torch.zeros(2, 4, dtype=torch.long)
            attention = torch.ones(2, 4)
            _, value = model(values, types, attention)
            value.sum().backward()
            gradient = model.input_projection.weight.grad
            return 0.0 if gradient is None else float(gradient.abs().sum())

        self.assertGreater(encoder_grad(1.0), 0.0)
        self.assertEqual(encoder_grad(0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
