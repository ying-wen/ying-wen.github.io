"""Exact algorithm-interface tests; Python 3.10+, standard library only.

Code: MIT. These are teaching kernels, not DQN/PPO/SAC training runners.
Commands: python algorithm_testing_lab.py test | demo
All output goes to stdout/stderr. No files are created or overwritten.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
import random
import unittest


def finite(*values):
    if not all(math.isfinite(value) for value in values):
        raise ValueError("non-finite input")


def probability(value):
    finite(value)
    if not 0 <= value <= 1:
        raise ValueError("probability/discount must be in [0, 1]")


# BEGIN boundaries
def td_target(reward, discount, next_value, terminated=False):
    finite(reward)
    probability(discount)
    # Branch before reading a terminal value: 0 * NaN is still NaN.
    if terminated or discount == 0:
        return reward
    finite(next_value)
    return reward + discount * next_value


@dataclass(frozen=True)
class GaeRow:
    reward: float
    value: float
    next_value: float | None
    terminated: bool = False
    # Does the NEXT ROW belong to this same trajectory segment?
    # False at a reset, an unconnected rollout cut, or the end of available data.
    trace_continues: bool = False


def gae(rows, discount=0.9, trace_decay=0.8, tail_advantage=0.0):
    probability(discount)
    probability(trace_decay)
    finite(tail_advantage)
    future, result = tail_advantage, []
    for row in reversed(rows):
        finite(row.value)
        delta = td_target(row.reward, discount, row.next_value,
                          row.terminated) - row.value
        connected = row.trace_continues and not row.terminated
        advantage = delta + discount * trace_decay * future if connected else delta
        result.append(advantage)
        future = advantage
    return list(reversed(result))
# END boundaries


# BEGIN policy_targets
def distribution(values):
    if not values:
        raise ValueError("empty policy")
    for value in values:
        probability(value)
    if not math.isclose(sum(values), 1.0, abs_tol=1e-12, rel_tol=0):
        raise ValueError("policy probabilities must sum to one")


def importance_ratio(target, behavior, action):
    distribution(target)
    distribution(behavior)
    if len(target) != len(behavior):
        raise ValueError("action spaces differ")
    if any(p > 0 and mu == 0 for p, mu in zip(target, behavior)):
        raise ValueError("target policy lacks behavior support")
    if not 0 <= action < len(target) or behavior[action] == 0:
        raise ValueError("recorded action is impossible under behavior policy")
    return target[action] / behavior[action]


def ppo_surrogate(ratio, advantage, clip=0.2):
    finite(ratio, advantage, clip)
    if ratio < 0 or not 0 <= clip < 1:
        raise ValueError("invalid PPO ratio or clip")
    clipped = max(1 - clip, min(1 + clip, ratio))
    # Objective to MAXIMIZE; a minimization loss negates this quantity.
    return min(ratio * advantage, clipped * advantage)


def double_q_target(reward, discount, online_next, target_next, terminated=False):
    if terminated or discount == 0:
        return td_target(reward, discount, None, terminated)
    if not online_next or len(online_next) != len(target_next):
        raise ValueError("invalid Q vectors")
    finite(*online_next, *target_next)
    selected = max(range(len(online_next)), key=online_next.__getitem__)
    return td_target(reward, discount, target_next[selected])
# END policy_targets


# BEGIN options
def option_backup(rewards, continuations, next_value, task_terminal=False):
    if not rewards or len(rewards) != len(continuations):
        raise ValueError("one continuation is required per primitive transition")
    total, product = 0.0, 1.0
    for reward, continuation in zip(rewards, continuations):
        finite(reward)
        probability(continuation)
        total += product * reward
        product *= continuation
    # Option termination alone does not end the task.
    if not task_terminal and product != 0:
        finite(next_value)
        total += product * next_value
    return total


def reward_rate(reward_totals, durations):
    if not durations or len(reward_totals) != len(durations):
        raise ValueError("unaligned option outcomes")
    finite(*reward_totals, *durations)
    if any(duration <= 0 for duration in durations):
        raise ValueError("duration must be positive")
    return sum(reward_totals) / sum(durations)
# END options


# BEGIN checkpoint
class TinyProcess:
    """A synthetic online predictor with explicitly owned persistent state.

    Its purpose is checkpoint equivalence, not control performance. The trace,
    momentum, normalization and environment all affect subsequent updates.
    """
    def __init__(self, seed=7):
        self.weight = 0.1
        self.velocity = 0.0
        self.trace = 0.0
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.world = 1.0
        self.time = 0
        self.environment_rng = random.Random(seed)
        self.probe_rng = random.Random(seed + 1000)

    def step(self):
        # Only past normalization statistics are read for this prediction.
        scale = math.sqrt(self.m2 / self.count + 1) if self.count else 1.0
        feature = (self.world - self.mean) / scale
        feature *= -1 if self.probe_rng.random() < 0.5 else 1
        prediction = self.weight * feature
        target = 0.5 * feature + self.environment_rng.uniform(-0.1, 0.1)
        error = target - prediction
        self.trace = 0.4 * self.trace + feature
        self.velocity = 0.6 * self.velocity + error * self.trace
        self.weight += 0.01 * self.velocity
        self.count += 1
        delta = self.world - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (self.world - self.mean)
        self.world = 0.7 * self.world + self.environment_rng.uniform(-1, 1)
        self.time += 1
        return {"time": self.time, "prediction": prediction,
                "target": target, "weight": self.weight}

    def snapshot(self):
        fields = ("weight", "velocity", "trace", "count", "mean", "m2",
                  "world", "time")
        return {"version": 1, "state": {key: getattr(self, key) for key in fields},
                "environment_rng": self.environment_rng.getstate(),
                "probe_rng": self.probe_rng.getstate()}

    @classmethod
    def restore(cls, snapshot):
        required = {"weight", "velocity", "trace", "count", "mean", "m2",
                    "world", "time"}
        if snapshot["version"] != 1 or set(snapshot["state"]) != required:
            raise ValueError("incomplete or incompatible checkpoint")
        finite(*snapshot["state"].values())
        instance = cls(0)
        for key, value in snapshot["state"].items():
            setattr(instance, key, value)
        # JSON turns tuples into lists. random.setstate expects tuples.
        def tuples(value):
            return tuple(tuples(x) for x in value) if isinstance(value, (list, tuple)) else value
        instance.environment_rng.setstate(tuples(snapshot["environment_rng"]))
        instance.probe_rng.setstate(tuples(snapshot["probe_rng"]))
        return instance
# END checkpoint


class AlgorithmTests(unittest.TestCase):
    def test_terminal_does_not_read_invalid_next_value(self):
        self.assertEqual(td_target(1, 0.9, float("nan"), True), 1)

    def test_external_truncation_bootstraps_final_observation(self):
        self.assertEqual(td_target(1, 0.9, 3), 3.7)
        self.assertNotEqual(td_target(1, 0.9, 3), td_target(1, 0.9, 100))

    def test_zero_discount_short_circuits_bootstrap(self):
        self.assertEqual(td_target(2, 0, None), 2)

    def test_invalid_live_value_is_rejected(self):
        with self.assertRaises(ValueError):
            td_target(1, 0.9, float("nan"))

    def test_discount_bounds(self):
        for gamma in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                td_target(1, gamma, 3)

    def test_gae_truncation_bootstraps_but_stops_recurrence(self):
        rows = [GaeRow(1, 2, 3), GaeRow(100, 0, None, True)]
        self.assertAlmostEqual(gae(rows)[0], 1.7)
        self.assertEqual(gae(rows)[1], 100)

    def test_gae_connected_trace_propagates(self):
        rows = [GaeRow(1, 2, 3, trace_continues=True), GaeRow(2, 3, None, True)]
        self.assertAlmostEqual(gae(rows)[0], 0.98)

    def test_gae_terminal_stops_even_if_trace_flag_is_true(self):
        self.assertEqual(gae([GaeRow(1, 0, None, True, True)], tail_advantage=99), [1])

    def test_gae_zero_lambda_is_td_error(self):
        self.assertAlmostEqual(gae([GaeRow(1, 2, 3, False, True)], trace_decay=0)[0], 1.7)

    def test_gae_explicit_connected_tail(self):
        self.assertAlmostEqual(gae([GaeRow(1, 2, 3, False, True)], tail_advantage=2)[0], 3.14)

    def test_importance_ratio_hand_value(self):
        self.assertAlmostEqual(importance_ratio([0.8, 0.2], [0.4, 0.6], 0), 2)

    def test_importance_ratio_zero_target(self):
        self.assertEqual(importance_ratio([0, 1], [0.4, 0.6], 0), 0)

    def test_importance_support_is_a_global_condition(self):
        with self.assertRaises(ValueError):
            importance_ratio([0.2, 0.8], [0, 1], 1)

    def test_recorded_action_must_have_positive_probability(self):
        with self.assertRaises(ValueError):
            importance_ratio([0, 1], [0, 1], 0)

    def test_importance_expectation_identity(self):
        target, behavior, values = [0.8, 0.2], [0.4, 0.6], [3, -2]
        weighted = sum(behavior[a] * importance_ratio(target, behavior, a) * values[a]
                       for a in range(2))
        self.assertAlmostEqual(weighted, sum(p * v for p, v in zip(target, values)))

    def test_invalid_probability_vector(self):
        with self.assertRaises(ValueError):
            importance_ratio([0.8, 0.8], [0.4, 0.6], 0)

    def test_ppo_positive_advantage_upper_clip(self):
        self.assertAlmostEqual(ppo_surrogate(1.4, 2), 2.4)

    def test_ppo_negative_advantage_low_ratio_clip(self):
        self.assertAlmostEqual(ppo_surrogate(0.6, -2), -1.6)

    def test_ppo_negative_advantage_high_ratio_not_clipped(self):
        self.assertAlmostEqual(ppo_surrogate(1.4, -2), -2.8)

    def test_ppo_zero_advantage(self):
        self.assertEqual(ppo_surrogate(100, 0), 0)

    def test_double_q_selects_online_evaluates_target(self):
        self.assertEqual(double_q_target(1, 0.9, [3, 2], [4, 20]), 4.6)

    def test_double_q_terminal_skips_both_networks(self):
        self.assertEqual(double_q_target(1, 0.9, [], [], True), 1)

    def test_option_duration_one_reduces_to_td(self):
        self.assertEqual(option_backup([1], [0.9], 3), td_target(1, 0.9, 3))

    def test_option_known_two_step_return(self):
        self.assertAlmostEqual(option_backup([1, 2], [0.9, 0.9], 3), 5.23)

    def test_option_variable_continuation(self):
        self.assertAlmostEqual(option_backup([1, 2], [0.5, 0.2], 3), 2.3)

    def test_option_termination_does_not_end_task(self):
        self.assertAlmostEqual(option_backup([1, 2], [0.9, 0.9], None, True), 2.8)
        self.assertAlmostEqual(option_backup([1, 2], [0.9, 0.9], 3, False), 5.23)

    def test_option_zero_continuation_ignores_later_rewards(self):
        self.assertEqual(option_backup([1, 100], [0, 1], None), 1)

    def test_option_alignment_rejected(self):
        with self.assertRaises(ValueError):
            option_backup([1, 2], [0.9], 3)

    def test_random_duration_is_not_average_duration(self):
        exact = 0.5 * 0.9 * 2 + 0.5 * 0.9 ** 3 * 6
        shortcut = 0.9 ** 2 * 4
        self.assertAlmostEqual(exact, 3.087)
        self.assertAlmostEqual(shortcut, 3.24)
        self.assertNotEqual(exact, shortcut)

    def test_reward_rate_weights_primitive_time(self):
        self.assertAlmostEqual(reward_rate([2, 9], [1, 9]), 1.1)
        self.assertNotEqual(reward_rate([2, 9], [1, 9]), (2 / 1 + 9 / 9) / 2)

    def test_nonpositive_duration_rejected(self):
        with self.assertRaises(ValueError):
            reward_rate([2], [0])

    def test_checkpoint_json_roundtrip_exact_continuation(self):
        original = TinyProcess(8)
        for _ in range(11):
            original.step()
        saved = json.loads(json.dumps(original.snapshot(), allow_nan=False))
        restored = TinyProcess.restore(saved)
        self.assertEqual([original.step() for _ in range(20)],
                         [restored.step() for _ in range(20)])
        self.assertEqual(original.snapshot(), restored.snapshot())

    def test_checkpoint_missing_optimizer_state_rejected(self):
        saved = TinyProcess().snapshot()
        del saved["state"]["velocity"]
        with self.assertRaises(ValueError):
            TinyProcess.restore(saved)

    def test_checkpoint_lost_rng_changes_future(self):
        original = TinyProcess(8)
        for _ in range(11):
            original.step()
        restored = TinyProcess.restore(original.snapshot())
        restored.environment_rng = random.Random(0)
        self.assertNotEqual(original.step(), restored.step())

    def test_diagnostic_branch_cannot_mutate_original(self):
        original = TinyProcess(9)
        saved = deepcopy(original.snapshot())
        branch = TinyProcess.restore(saved)
        for _ in range(30):
            branch.step()
        self.assertEqual(original.snapshot(), saved)

    def test_semi_gradient_freezes_bootstrap_target(self):
        weight, feature, target = 0.3, 2.0, 1.4
        loss = lambda w: 0.5 * (target - w * feature) ** 2
        epsilon = 1e-5
        numeric = (loss(weight + epsilon) - loss(weight - epsilon)) / (2 * epsilon)
        self.assertAlmostEqual(numeric, -(target - weight * feature) * feature, places=9)


def demo():
    process = TinyProcess(8)
    for _ in range(11):
        process.step()
    restored = TinyProcess.restore(json.loads(json.dumps(process.snapshot())))
    equal = [process.step() for _ in range(20)] == [restored.step() for _ in range(20)]
    return {"scope": "exact teaching kernels; no deep-RL training claim",
            "boundary": {"terminated_target": td_target(1, 0.9, None, True),
                         "external_truncation_target": td_target(1, 0.9, 3),
                         "gae_no_reset_leak": gae([GaeRow(1, 2, 3), GaeRow(100, 0, None, True)])},
            "importance_ratio": importance_ratio([0.8, 0.2], [0.4, 0.6], 0),
            "ppo_positive_negative": [ppo_surrogate(1.4, 2), ppo_surrogate(0.6, -2)],
            "double_q_target": double_q_target(1, 0.9, [3, 2], [4, 20]),
            "option_two_step": option_backup([1, 2], [0.9, 0.9], 3),
            "duration_correlation": {"exact": 0.5 * 0.9 * 2 + 0.5 * 0.9 ** 3 * 6,
                                     "invalid_mean_shortcut": 0.9 ** 2 * 4},
            "primitive_reward_rate": reward_rate([2, 9], [1, 9]),
            "checkpoint_identical_next_20_events": equal}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("test", "demo"))
    args = parser.parse_args()
    if args.command == "test":
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(AlgorithmTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    print(json.dumps(demo(), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
