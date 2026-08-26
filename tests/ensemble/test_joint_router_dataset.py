from __future__ import annotations

from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from farmer_ensemble.joint_router_dataset import (
    DECISION_IDS,
    DECISION_NAMES,
    decision_feature_names,
    record_to_rows,
)


def observation() -> dict:
    board = [[None for _ in range(10)] for _ in range(10)]
    return {
        "player": 0,
        "step": 0,
        "day": 0,
        "hour": 0,
        "farms": [
            {
                "money": 3000,
                "tiles": board,
                "farmer": [4, 4],
                "hands": [[5, 4]],
                "unlocked_quadrants": ["NW"],
                "hires_today": 1,
            },
            {
                "money": 3000,
                "tiles": board,
                "farmer": [4, 4],
                "hands": [],
                "unlocked_quadrants": ["NW"],
                "hires_today": 0,
            },
        ],
        "private": {
            "shed": {},
            "seeds": {},
            "inventories": [{}, {}],
        },
        "market": {"prices": {}, "inventory": {}},
        "town": {"unlocked_shops": []},
    }


class JointRouterDatasetTests(unittest.TestCase):
    def test_vocabulary_covers_unit_and_market_semantics(self) -> None:
        self.assertEqual(len(DECISION_NAMES), 25)
        self.assertIn("PLACE", DECISION_IDS)
        self.assertIn("BUY_PRODUCT", DECISION_IDS)
        self.assertIn("BUY_LAND", DECISION_IDS)

    def test_rows_have_one_market_terminator_without_padding(self) -> None:
        record = {
            "episode_id": "episode-1",
            "acting_seat": 0,
            "seed": 1,
            "observation": observation(),
            "action": {
                "farmer": ["PLANT", "WHEAT"],
                "hands": [["WEST"]],
                "market": [["BUY_SEED", "WHEAT", 2], ["HIRE"]],
            },
        }
        rows = record_to_rows(record)
        self.assertEqual(len(rows), 5)
        self.assertEqual(
            [DECISION_NAMES[label] for _, label, _ in rows],
            ["PLANT", "WEST", "BUY_SEED", "HIRE", "NO_ORDER"],
        )
        self.assertTrue(all(len(features) == len(decision_feature_names()) for features, _, _ in rows))

    def test_duplicate_market_orders_are_rejected(self) -> None:
        record = {
            "episode_id": "episode-1",
            "acting_seat": 0,
            "seed": 1,
            "observation": observation(),
            "action": {
                "farmer": ["PASS"],
                "hands": [["PASS"]],
                "market": [["BUY_SEED", "WHEAT", 1], ["BUY_SEED", "WHEAT", 1]],
            },
        }
        with self.assertRaisesRegex(ValueError, "duplicate market order"):
            record_to_rows(record)


if __name__ == "__main__":
    unittest.main()
