from __future__ import annotations

import json
import importlib.util
import random
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from farmer_rl.actions import CandidateGenerator, JointActionCodec
from farmer_rl.collector import collect_episode
from farmer_rl.environment import KaggricultureEnv, pass_action
from farmer_rl.errors import InvalidActionError, SeatSafetyError
from farmer_rl.opponents import OpponentPool, OpponentSpec
from farmer_rl.tokenizer import FEATURE_DIM, ObservationTokenizer, TILE_KINDS
from farmer_rl.trajectory import EpisodeTrajectory, Transition


def observation(seat: int, *, step: int = 0, hands: int = 0) -> dict:
    empty = [[None for _ in range(10)] for _ in range(10)]
    boards = [deepcopy(empty), deepcopy(empty)]
    boards[0][0][0] = {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": 0,
        "watered_today": False,
        "yield_units": 2,
    }
    boards[1][0][0] = "LOCKED"
    farms = []
    for index in (0, 1):
        farms.append(
            {
                "money": 3000 + 100 * index,
                "tiles": boards[index],
                "farmer": [0, 0],
                "hands": [[1, 0] for _ in range(hands)],
                "unlocked_quadrants": ["NW"],
                "hires_today": hands,
            }
        )
    return {
        "player": seat,
        "step": step,
        "day": step // 24,
        "hour": step % 24,
        "farms": farms,
        "private": {
            "shed": {"WHEAT": 3, "FERTILIZER": 1},
            "seeds": {"WHEAT": 2},
            "inventories": [{} for _ in range(1 + hands)],
        },
        "market": {
            "inventory": {"WHEAT": 100},
            "prices": {"WHEAT": 25},
        },
        "town": {"unlocked_shops": ["BAKERY"]},
    }


class FakeOfficialEnvironment:
    def __init__(self) -> None:
        self.turn = 0

    def reset(self, num_agents: int):
        assert num_agents == 2
        return [SimpleNamespace(observation=observation(seat, step=0), reward=0, status="ACTIVE") for seat in (0, 1)]

    def step(self, actions):
        assert len(actions) == 2
        self.turn += 1
        status = "DONE" if self.turn >= 1 else "ACTIVE"
        return [
            SimpleNamespace(observation=observation(seat, step=self.turn), reward=1 - 2 * seat, status=status)
            for seat in (0, 1)
        ]


def fake_make(name: str, configuration: dict, debug: bool):
    assert name == "kaggriculture"
    assert configuration["episodeSteps"] == 720
    assert debug is False
    return FakeOfficialEnvironment()


class EnvironmentTests(unittest.TestCase):
    def test_wrapper_preserves_each_acting_seat(self):
        env = KaggricultureEnv(make_fn=fake_make)
        initial = env.reset(seed=7)
        self.assertEqual(initial[0]["player"], 0)
        self.assertEqual(initial[1]["player"], 1)
        result = env.step({0: pass_action(), 1: pass_action()})
        self.assertTrue(result.terminated)
        self.assertEqual(result.rewards, {0: 1.0, 1: -1.0})

    def test_wrapper_rejects_cross_seat_state(self):
        states = [
            SimpleNamespace(observation=observation(1)),
            SimpleNamespace(observation=observation(0)),
        ]
        with self.assertRaises(SeatSafetyError):
            KaggricultureEnv._seat_observations(states)

    def test_action_envelope_checks_current_hand_count(self):
        env = KaggricultureEnv(make_fn=fake_make)
        env.reset()
        with self.assertRaises(InvalidActionError):
            env.step({0: pass_action(1), 1: pass_action()})


class TrajectoryTests(unittest.TestCase):
    def _transition(self, seat: int, step: int = 0) -> Transition:
        return Transition(
            episode_id="episode",
            step=step,
            acting_seat=seat,
            observation=observation(seat, step=step),
            action=pass_action(),
            reward=0.0,
            next_observation=observation(seat, step=step + 1),
            terminated=False,
            opponent_id=f"policy-{1-seat}",
            policy_id=f"policy-{seat}",
        )

    def test_transition_refuses_wrong_player(self):
        value = self._transition(0).to_dict()
        value["observation"]["player"] = 1
        with self.assertRaises(SeatSafetyError):
            Transition.from_dict(value)

    def test_per_seat_steps_must_increase_even_when_interleaved(self):
        episode = EpisodeTrajectory("episode")
        episode.append(self._transition(0, 1))
        episode.append(self._transition(1, 1))
        with self.assertRaises(ValueError):
            episode.append(self._transition(0, 1))

    def test_collector_is_closed_loop_and_memory_only(self):
        def policy(obs):
            obs["private"]["shed"]["WHEAT"] = 999  # must not mutate recorded observation
            return pass_action()

        trajectory = collect_episode(
            KaggricultureEnv(make_fn=fake_make),
            {0: policy, 1: policy},
            policy_ids={0: "left", 1: "right"},
            seed=9,
        )
        self.assertEqual(len(trajectory.transitions), 2)
        self.assertEqual(trajectory.transitions[0].observation["private"]["shed"]["WHEAT"], 3)
        self.assertEqual({item.acting_seat for item in trajectory.transitions}, {0, 1})


class TokenAndActionTests(unittest.TestCase):
    def test_tokenizer_role_orders_own_farm_first(self):
        tokenizer = ObservationTokenizer(max_tokens=320)
        seat_zero = tokenizer.tokenize(observation(0))
        seat_one = tokenizer.tokenize(observation(1))
        self.assertEqual(seat_zero.shape, (320, FEATURE_DIM))
        self.assertEqual(seat_zero.values[1][1], 1.0)
        self.assertAlmostEqual(seat_zero.values[1][4], TILE_KINDS["PLANT"] / 5)
        self.assertAlmostEqual(seat_one.values[1][4], TILE_KINDS["LOCKED"] / 5)
        self.assertEqual(sum(seat_zero.attention_mask), sum(seat_one.attention_mask))

    def test_candidates_mask_and_joint_decode(self):
        obs = observation(0)
        generator = CandidateGenerator(capacity=32)
        farmer = generator.unit_candidates(obs, 0)
        operations = {candidate.action[0] for candidate in farmer.candidates}
        self.assertIn("WATER", operations)
        self.assertIn("HARVEST", operations)
        self.assertEqual(len(farmer.mask), 32)
        codec = JointActionCodec(generator, max_hands=2)
        mask = codec.mask(obs)
        self.assertEqual(len(mask), (1 + 2 + 10) * 32)
        action = codec.decode(obs, [0] * codec.slots)
        self.assertEqual(action, pass_action())

    def test_market_compiler_does_not_oversell_repeated_candidates(self):
        obs = observation(0)
        generator = CandidateGenerator(capacity=32)
        codec = JointActionCodec(generator, max_hands=0, max_orders=2)
        candidate_sets = codec.candidates(obs)
        sell_indices = []
        for candidate_set in candidate_sets[-2:]:
            sell_indices.append(next(i for i, item in enumerate(candidate_set.candidates) if item.action[:2] == ("SELL", "WHEAT")))
        action = codec.decode(obs, [0, *sell_indices])
        self.assertEqual(sum(order[2] for order in action["market"] if order[:2] == ["SELL", "WHEAT"]), 3)


class OpponentAndConfigTests(unittest.TestCase):
    def test_pfsp_prefers_near_even_opponent(self):
        pool = OpponentPool(
            (
                OpponentSpec("easy", "test"),
                OpponentSpec("even", "test"),
            )
        )
        for _ in range(10):
            pool.record("easy", 1.0)
        weights = pool.sampling_weights()
        self.assertGreater(weights["even"], weights["easy"])
        self.assertIn(pool.sample(random.Random(3)).opponent_id, {"easy", "even"})

    def test_json_configs_parse(self):
        root = Path(__file__).resolve().parents[2]
        for filename in ("ppo.json", "population.json", "data_manifest.schema.json", "data_manifest.example.json"):
            with (root / "configs" / "rl" / filename).open(encoding="utf-8") as handle:
                self.assertIsInstance(json.load(handle), dict)

    @unittest.skipUnless(importlib.util.find_spec("jsonschema"), "jsonschema is optional")
    def test_example_manifest_matches_schema(self):
        import jsonschema

        root = Path(__file__).resolve().parents[2]
        with (root / "configs" / "rl" / "data_manifest.schema.json").open(encoding="utf-8") as handle:
            schema = json.load(handle)
        with (root / "configs" / "rl" / "data_manifest.example.json").open(encoding="utf-8") as handle:
            example = json.load(handle)
        jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(example)


if __name__ == "__main__":
    unittest.main()
