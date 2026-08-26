from __future__ import annotations

import time
import unittest

from farmer_eval.benchmark import SafePolicy, summarize_games


def observation(seat: int, *, hands: int = 0) -> dict:
    farm = {
        "money": 3000,
        "tiles": [[None for _ in range(10)] for _ in range(10)],
        "farmer": [4, 4],
        "hands": [[4, 4] for _ in range(hands)],
        "unlocked_quadrants": ["NW"],
        "hires_today": hands,
    }
    return {
        "player": seat,
        "farms": [dict(farm), dict(farm)],
        "private": {"shed": {}, "seeds": {}, "inventories": [{} for _ in range(hands + 1)]},
    }


class SafePolicyTests(unittest.TestCase):
    def test_exception_and_malformed_actions_become_shape_safe_pass(self):
        for policy in (lambda obs: 1 / 0, lambda obs: {"farmer": ["PASS"], "hands": []}):
            wrapped = SafePolicy(policy, timeout=0.1, max_faults=1)
            action = wrapped.act(observation(0, hands=2))
            self.assertEqual(action, {"farmer": ["PASS"], "hands": [["PASS"], ["PASS"]], "market": []})
            self.assertTrue(wrapped.faulted)

    def test_timeout_disables_policy_for_remainder_of_match(self):
        def slow(obs):
            time.sleep(0.1)
            return {"farmer": ["PASS"], "hands": [], "market": []}

        wrapped = SafePolicy(slow, timeout=0.01, max_faults=2)
        wrapped.act(observation(0))
        started = time.perf_counter()
        wrapped.act(observation(0))
        self.assertLess(time.perf_counter() - started, 0.05)
        self.assertEqual(wrapped.timeouts, 1)
        self.assertEqual(wrapped.disabled_calls, 1)

    def test_collapse_diagnostics_detect_duplicate_orders_and_idle_streak(self):
        def duplicate(obs):
            return {
                "farmer": ["PASS"],
                "hands": [],
                "market": [["BUY_SEED", "CARROT", 1] for _ in range(10)],
            }

        wrapped = SafePolicy(duplicate, timeout=0.1, max_faults=2)
        wrapped.act(observation(0))
        metrics = wrapped.metrics()
        self.assertEqual(metrics["market_orders"], 10)
        self.assertEqual(metrics["duplicate_market_orders"], 9)
        self.assertAlmostEqual(metrics["duplicate_market_order_rate"], 0.9)
        self.assertEqual(metrics["max_idle_pass_streak"], 0)
        self.assertEqual(metrics["unique_joint_actions"], 1)


class SummaryTests(unittest.TestCase):
    @staticmethod
    def game(seat: int, outcome: float, difference: float) -> dict:
        policy = {"faulted": False, "action_distribution": {"PASS": 2}, "_latencies": [0.001, 0.002]}
        return {
            "candidate": "candidate",
            "opponent": "opponent",
            "seed": 7,
            "candidate_seat": seat,
            "score_difference": difference,
            "outcome": outcome,
            "engine_error": None,
            "candidate_policy": policy,
        }

    def test_paired_seed_and_seat_metrics(self):
        summary = summarize_games([self.game(0, 1.0, 10.0), self.game(1, 0.5, 0.0)])
        self.assertEqual(summary["overall"]["wins"], 1)
        self.assertEqual(summary["overall"]["draws"], 1)
        self.assertEqual(summary["paired_seed_groups"], 1)
        self.assertEqual(summary["paired_seed_mean_outcome"], 0.75)
        self.assertEqual(set(summary["by_candidate_seat"]), {"0", "1"})


if __name__ == "__main__":
    unittest.main()
