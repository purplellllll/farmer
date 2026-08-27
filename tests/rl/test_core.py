from __future__ import annotations

import json
import importlib.util
import random
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from farmer_rl.actions import CandidateGenerator, JointActionCodec
from farmer_rl.bc import _validation_group, iter_bc_records
from farmer_rl.collector import collect_episode
from farmer_rl.environment import KaggricultureEnv, pass_action
from farmer_rl.errors import InvalidActionError, SeatSafetyError
from farmer_rl.opponents import OpponentPool, OpponentSpec
from farmer_rl.native_ppo import _potential, _terminal_reward, _update
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
    def test_official_structs_are_recursively_detached(self):
        class Struct(dict):
            pass

        value = Struct({"nested": Struct({"items": [Struct({"x": 1})]})})
        plain = KaggricultureEnv._plain_value(value)
        self.assertIs(type(plain), dict)
        self.assertIs(type(plain["nested"]), dict)
        self.assertIs(type(plain["nested"]["items"][0]), dict)

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

    def test_close_releases_official_environment_history(self):
        env = KaggricultureEnv(make_fn=fake_make)
        env.reset()
        self.assertIsInstance(env.raw_environment, FakeOfficialEnvironment)
        env.close()
        self.assertEqual(env._last_observations, {})
        with self.assertRaises(RuntimeError):
            _ = env.raw_environment


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


class BehaviourCloneDataTests(unittest.TestCase):
    def test_minimal_curriculum_schema_and_group_split_are_seat_safe(self):
        records = []
        for seat in (0, 1):
            records.append(
                {
                    "episode_id": "episode-shared",
                    "step": 7,
                    "acting_seat": seat,
                    "observation": observation(seat, step=7),
                    "action": pass_action(),
                    "seed": 99,
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "curriculum.jsonl"
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            loaded = list(iter_bc_records([path]))
        self.assertEqual([item["acting_seat"] for item in loaded], [0, 1])
        assignments = {
            _validation_group(item, fraction=0.5, split_seed=1234)
            for item in loaded
        }
        self.assertEqual(len(assignments), 1)


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

    def test_market_compiler_deduplicates_repeated_purchases(self):
        obs = observation(0)
        generator = CandidateGenerator(capacity=64)
        codec = JointActionCodec(generator, max_hands=0, max_orders=10)
        candidate_sets = codec.candidates(obs)
        buy_indices = [
            next(
                i
                for i, item in enumerate(candidate_set.candidates)
                if item.action == ("BUY_SEED", "WHEAT", 1)
            )
            for candidate_set in candidate_sets[-10:]
        ]
        action = codec.decode(obs, [0, *buy_indices])
        self.assertEqual(action["market"], [["BUY_SEED", "WHEAT", 1]])

    def test_prefix_conditioned_selection_masks_duplicate_market_orders(self):
        obs = observation(0)
        codec = JointActionCodec(CandidateGenerator(capacity=64), max_hands=0, max_orders=3)

        def chooser(_slot_index, candidate_set, valid_mask):
            for index, candidate in enumerate(candidate_set.candidates):
                if candidate.action == ("BUY_SEED", "WHEAT", 1) and valid_mask[index]:
                    return index
            return 0

        indices, masks = codec.select(obs, chooser)
        action = codec.decode(obs, indices)
        rows = [masks[index * 64 : (index + 1) * 64] for index in range(codec.slots)]
        repeated_buy_index = next(
            index
            for index, candidate in enumerate(codec.candidates(obs)[-2].candidates)
            if candidate.action == ("BUY_SEED", "WHEAT", 1)
        )
        self.assertEqual(action["market"], [["BUY_SEED", "WHEAT", 1]])
        self.assertEqual(rows[-2][repeated_buy_index], 0)
        self.assertEqual(sum(rows[-1]), 1)

    def test_prefix_conditioned_selection_reserves_shared_seeds(self):
        obs = observation(0, hands=3)
        obs["farms"][0]["hands"] = [[1, 0], [2, 0], [3, 0]]
        codec = JointActionCodec(CandidateGenerator(capacity=64), max_hands=3, max_orders=1)

        def chooser(_slot_index, candidate_set, valid_mask):
            for index, candidate in enumerate(candidate_set.candidates):
                if candidate.action == ("PLANT", "WHEAT") and valid_mask[index]:
                    return index
            return 0

        indices, _ = codec.select(obs, chooser)
        action = codec.decode(obs, indices)
        self.assertEqual(action["hands"][:2], [["PLANT", "WHEAT"], ["PLANT", "WHEAT"]])
        self.assertEqual(action["hands"][2], ["PASS"])

    def test_hire_candidate_stops_at_bounded_hand_count(self):
        obs = observation(0, hands=8)
        candidates = CandidateGenerator(capacity=64).market_candidates(obs, 0).candidates
        self.assertNotIn("HIRE", {candidate.action[0] for candidate in candidates})


class OpponentAndConfigTests(unittest.TestCase):
    def test_productive_shaping_preserves_planted_capital_and_values_maturity(self):
        seed_state = observation(0)
        seed_state["farms"][0]["money"] = seed_state["farms"][1]["money"]
        seed_state["farms"][0]["tiles"][0][0] = None
        seed_state["private"]["shed"] = {}
        seed_state["private"]["seeds"] = {"WHEAT": 1}

        planted_state = deepcopy(seed_state)
        planted_state["private"]["seeds"] = {}
        planted_state["farms"][0]["tiles"][0][0] = {
            "kind": "PLANT",
            "crop": "WHEAT",
            "planted_day": 0,
            "watered_today": False,
            "consecutive_unwatered": 1,
            "yield_units": 0,
        }
        mature_state = deepcopy(planted_state)
        mature_state["day"] = 2
        mature_state["farms"][0]["tiles"][0][0].update(
            watered_today=True,
            consecutive_unwatered=0,
            yield_units=2,
        )

        self.assertLess(_potential(planted_state, 0), _potential(seed_state, 0))
        self.assertAlmostEqual(
            _potential(planted_state, 0, profile="production_cycle_v2"),
            _potential(seed_state, 0, profile="production_cycle_v2"),
        )
        self.assertGreater(
            _potential(mature_state, 0, profile="production_cycle_v2"),
            _potential(planted_state, 0, profile="production_cycle_v2"),
        )

    def test_shaping_profile_is_validated(self):
        with self.assertRaises(ValueError):
            _potential(observation(0), 0, profile="unknown")
        with self.assertRaises(ValueError):
            _potential(observation(0), 0, scale=0)

    def test_terminal_reward_preserves_win_order_and_dense_loss_margin(self):
        close_outcome, close_reward = _terminal_reward(
            -100, score_coefficient=0.25, score_scale=1000
        )
        bad_outcome, bad_reward = _terminal_reward(
            -3000, score_coefficient=0.25, score_scale=1000
        )
        win_outcome, win_reward = _terminal_reward(
            1, score_coefficient=0.25, score_scale=1000
        )
        self.assertEqual((close_outcome, bad_outcome, win_outcome), (0.0, 0.0, 1.0))
        self.assertGreater(close_reward, bad_reward)
        self.assertGreater(win_reward, close_reward)

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
        for filename in (
            "ppo.json",
            "local_4060.json",
            "local_4060_recovery_v2.json",
            "local_4060_recovery_v5.json",
            "cpu_recovery_v3.json",
            "cpu_recovery_v5.json",
            "cpu_v2.json",
            "cpu_v2_smoke.json",
            "population.json",
            "data_manifest.schema.json",
            "data_manifest.example.json",
        ):
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


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is optional")
class NativePpoStabilityTests(unittest.TestCase):
    def test_update_keeps_dropout_disabled_for_rollout_parity(self):
        import torch
        from torch import nn

        class TinyActorCritic(nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = nn.Parameter(torch.zeros(1, 1, 2))
                self.value = nn.Parameter(torch.zeros(1))
                self.dropout = nn.Dropout(0.9)

            def forward(self, values, _types, _attention, action_mask):
                logits = self.dropout(self.logits).expand(values.shape[0], -1, -1)
                return logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min), self.value.expand(values.shape[0])

        model = TinyActorCritic()
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        batch = {
            "tokens": torch.zeros(2, 1, 1),
            "types": torch.zeros(2, 1, dtype=torch.long),
            "attention": torch.ones(2, 1, dtype=torch.bool),
            "action_mask": torch.ones(2, 1, 2, dtype=torch.bool),
            "actions": torch.zeros(2, 1, dtype=torch.long),
            "policy_slot_mask": torch.ones(2, 1),
            "old_log_probability": torch.full((2,), -0.69314718),
            "old_value": torch.zeros(2),
            "advantage": torch.tensor([1.0, -1.0]),
            "return": torch.zeros(2),
        }

        metrics = _update(
            model,
            optimizer,
            batch,
            device=torch.device("cpu"),
            minibatch_size=2,
            epochs=1,
            clip_param=0.1,
            value_coeff=0.5,
            entropy_coeff=0.0,
            grad_clip=1.0,
            target_kl=1.0,
        )

        self.assertFalse(model.training)
        self.assertEqual(metrics["kl_early_stop"], 0.0)


if __name__ == "__main__":
    unittest.main()
