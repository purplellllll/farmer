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


if __name__ == "__main__":
    unittest.main()
