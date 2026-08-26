from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from farmer_ensemble.joint_policy import JointEnsemblePolicy
from farmer_ensemble.joint_router_dataset import DECISION_IDS
from tests.ensemble.test_joint_router_dataset import observation


class StubBundle:
    classes_ = np.arange(25)

    def predict_proba(self, X):
        values = np.full((len(X), 25), 1e-4)
        values[:, DECISION_IDS["BUY_SEED"]] = 0.5
        values[:, DECISION_IDS["PLANT"]] = 0.4
        values[:, DECISION_IDS["NO_ORDER"]] = 0.3
        return values / values.sum(axis=1, keepdims=True)


class JointPolicyTests(unittest.TestCase):
    def test_decoder_reserves_resources_and_has_no_duplicate_orders(self) -> None:
        obs = observation()
        obs["private"]["seeds"] = {"WHEAT": 1}
        obs["market"]["prices"] = {"WHEAT": 25, "FERTILIZER": 100}
        policy = JointEnsemblePolicy(StubBundle())
        action = policy(obs)
        self.assertEqual(len(action["hands"]), 1)
        self.assertEqual(len(action["market"]), len({tuple(order) for order in action["market"]}))
        planted = [value for value in (action["farmer"], *action["hands"]) if value[0] == "PLANT"]
        self.assertLessEqual(len(planted), 1)


if __name__ == "__main__":
    unittest.main()
