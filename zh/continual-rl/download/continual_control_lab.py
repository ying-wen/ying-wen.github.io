#!/usr/bin/env python3
"""Exact teaching examples for continual control, not a paper benchmark.

python3 continual_control_lab.py demo
python3 continual_control_lab.py test

Independent simulated worlds permit counterfactual calculations here. This does
not grant reset, cloning, full feedback, or future information to a real agent.
"""
import argparse
import itertools
import json
import math
import unittest
from dataclasses import dataclass, field


def check_horizon(horizon, allow_zero=True):
    if not isinstance(horizon, int) or horizon < (0 if allow_zero else 1):
        raise ValueError("horizon must be a nonnegative integer")


def discounted(rewards, gamma=1.0):
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must lie in [0, 1]")
    return sum(gamma ** t * r for t, r in enumerate(rewards))


# BEGIN objectives
def investment_returns(horizon, gamma=1.0):
    """Choose once: safe gives 1 forever; invest gives -2, then 2 forever."""
    check_horizon(horizon)
    safe = [1.0] * horizon
    invest = [-2.0] + [2.0] * (horizon - 1) if horizon else []
    return {"safe": discounted(safe, gamma), "invest": discounted(invest, gamma)}


def sigmoid(theta):
    if theta >= 0:
        return 1 / (1 + math.exp(-theta))
    z = math.exp(theta)
    return z / (1 + z)


def investment_objective(theta, horizon):
    p = sigmoid(theta)
    returns = investment_returns(horizon)
    objective = p * returns["invest"] + (1 - p) * returns["safe"]
    gradient = p * (1 - p) * (returns["invest"] - returns["safe"])
    return objective, gradient
# END objectives


# BEGIN learners
@dataclass
class GreedyLearner:
    alpha: float
    q: list = field(default_factory=lambda: [1.0, 0.5])
    updates: int = 0

    def __post_init__(self):
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must lie in [0, 1]")
        self.q = list(self.q)

    def action(self):
        # Tie convention matters: choose action 0 when the two values are equal.
        return max(range(2), key=lambda a: self.q[a])

    def observe(self, action, reward):
        self.q[action] += self.alpha * (reward - self.q[action])
        self.updates += 1


def reversal_lifetime(alpha, horizon=10):
    """After a change, reward(0)=0, reward(1)=1. No resets inside a lifetime."""
    check_horizon(horizon)
    learner = GreedyLearner(alpha)
    actions, rewards = [], []
    for _ in range(horizon):
        action = learner.action()
        reward = float(action == 1)
        actions.append(action)
        rewards.append(reward)
        learner.observe(action, reward)
    return {"actions": actions, "rewards": rewards, "total": sum(rewards),
            "final_q": learner.q, "updates": learner.updates}


def fixed_stream_fit(alpha, actions=(0, 1, 0, 1)):
    """Learner is a passive predictor; logged actions do not come from its policy."""
    learner = GreedyLearner(alpha)
    for action in actions:
        learner.observe(action, float(action == 1))
    return {"final_q": learner.q, "next_action": learner.action()}
# END learners


# BEGIN history
def history_values():
    """A fair bit is revealed once, then the current observation becomes blank."""
    frozen_action_zero = {"bit_0_then_blank": 1.0, "bit_1_then_blank": 0.0}
    memory_policy = {"bit_0_then_blank": 1.0, "bit_1_then_blank": 1.0}
    return {"observation": "blank", "frozen_action_zero": frozen_action_zero,
            "observation_mixture_for_action_zero": 0.5,
            "remembering_agent": memory_policy,
            "remembering_mean": 1.0, "memoryless_mean": 0.5}


def unpredictable_comparator(horizon=4, rule=lambda history: 0.5):
    """Each independent fair bit identifies that step's rewarding action.

    rule receives only earlier bits, which binary reward/action feedback reveals.
    It returns the probability of action 1 BEFORE the current bit is revealed.
    """
    check_horizon(horizon)
    if horizon > 12:
        raise ValueError("exact enumeration limited to 12 steps")
    total = 0.0
    for bits in itertools.product((0, 1), repeat=horizon):
        reward = 0.0
        for t, bit in enumerate(bits):
            p_one = rule(bits[:t])
            if not 0 <= p_one <= 1:
                raise ValueError("causal rule must return a probability")
            reward += p_one if bit else 1 - p_one
        total += 2 ** (-horizon) * reward
    return {"causal_expected_reward": total, "clairvoyant_reward": float(horizon),
            "dynamic_regret": horizon - total}
# END history


# BEGIN deviations
def validate_distribution(probabilities):
    if not probabilities or any(not math.isfinite(p) or p < 0 for p in probabilities):
        raise ValueError("invalid probability distribution")
    if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-12):
        raise ValueError("probabilities must sum to 1")


def segment_deviation(records, gamma=0.9):
    """One complete H-step segment: (action, reward, behavior_probs, target_probs).

    Probabilities must be conditional on the actual prefix at that step. The
    target is the declared deviation applied at that prefix, not an action label
    copied from another rollout. Positive means deviation return exceeds agent
    return (Eq. 1 convention in Elelimy et al., 2025).
    """
    if not records:
        raise ValueError("empty segment")
    weight, rewards = 1.0, []
    for action, reward, behavior, target in records:
        validate_distribution(behavior)
        validate_distribution(target)
        if len(behavior) != len(target) or not 0 <= action < len(behavior):
            raise ValueError("action dimensions differ")
        if any(q > 0 and b == 0 for b, q in zip(behavior, target)):
            raise ValueError("target action lacks behavior support")
        if behavior[action] == 0 or not math.isfinite(reward):
            raise ValueError("impossible logged action or nonfinite reward")
        weight *= target[action] / behavior[action]
        rewards.append(reward)
    realized_return = discounted(rewards, gamma)
    return (weight - 1) * realized_return


def deviation_estimate(records, horizon=1, gamma=0.9):
    """Use complete windows only. Overlapping windows are NOT independent runs."""
    check_horizon(horizon, allow_zero=False)
    if len(records) < horizon:
        raise ValueError("not enough data for a complete window")
    values = [segment_deviation(records[t:t + horizon], gamma)
              for t in range(len(records) - horizon + 1)]
    return sum(values) / len(values)


def exact_deviation_check(horizon=2, gamma=0.9):
    """Enumerate a one-state bandit: b(1)=.5, deviation(1)=.75, r=a."""
    check_horizon(horizon, allow_zero=False)
    if horizon > 12:
        raise ValueError("exact enumeration limited to 12 steps")
    expectation = 0.0
    for actions in itertools.product((0, 1), repeat=horizon):
        records = [(a, float(a), (0.5, 0.5), (0.25, 0.75)) for a in actions]
        expectation += 0.5 ** horizon * segment_deviation(records, gamma)
    known = discounted([0.75 - 0.5] * horizon, gamma)
    return {"IS_expectation": expectation, "known_difference": known}
# END deviations


# BEGIN irreversible
def trap_lifetime(risky_first, horizon=10):
    """Safe action yields 1; risky yields 4 ONCE and destroys all future reward."""
    check_horizon(horizon)
    destroyed = False
    rewards, states = [], []
    for t in range(horizon):
        states.append("destroyed" if destroyed else "healthy")
        if destroyed:
            reward = 0.0
        elif risky_first and t == 0:
            reward, destroyed = 4.0, True
        else:
            reward = 1.0
        rewards.append(reward)
    return {"rewards": rewards, "states": states, "total": sum(rewards),
            "destroyed": destroyed}


def trap_diagnostics(lifetime=100, local_horizon=6):
    check_horizon(lifetime, allow_zero=False)
    check_horizon(local_horizon, allow_zero=False)
    risky, safe = trap_lifetime(True, lifetime), trap_lifetime(False, lifetime)
    # Known-model diagnostic from each history actually visited by risky agent.
    # Only the initial healthy history admits a beneficial deviation.
    local_gap = (local_horizon - 4.0) / lifetime
    return {"risky_total": risky["total"], "safe_total": safe["total"],
            "from_initial_world_gap": safe["total"] - risky["total"],
            "mean_local_deviation_gap": local_gap,
            "post_destruction_deviation_gap": 0.0,
            "access": "exact known simulator, not estimated from a real counterfactual"}
# END irreversible


def demo():
    return {"horizon_reversal": {str(t): investment_returns(t) for t in (3, 4, 5)},
            "same_initial_policy": {str(a): reversal_lifetime(a) for a in (0.0, 0.1, 1.0)},
            "fixed_stream_predictor": fixed_stream_fit(1.0),
            "history_value": history_values(),
            "unpredictable_oracle": unpredictable_comparator(),
            "deviation_IS": exact_deviation_check(), "irreversible": trap_diagnostics()}


class ContinualControlTests(unittest.TestCase):
    def test_short_horizon_prefers_safe(self):
        self.assertEqual(investment_returns(3), {"safe": 3.0, "invest": 2.0})

    def test_equal_horizon(self):
        self.assertEqual(investment_returns(4), {"safe": 4.0, "invest": 4.0})

    def test_long_horizon_prefers_invest(self):
        self.assertEqual(investment_returns(5), {"safe": 5.0, "invest": 6.0})

    def test_empty_lifetime(self):
        self.assertEqual(investment_returns(0), {"safe": 0, "invest": 0})

    def test_zero_discount(self):
        self.assertEqual(investment_returns(10, 0), {"safe": 1.0, "invest": -2.0})

    def test_discount_threshold(self):
        value = investment_returns(200, 0.75)
        self.assertAlmostEqual(value["safe"], value["invest"])

    def test_finite_difference_objective(self):
        theta, eps = 0.3, 1e-6
        for horizon in (3, 4, 7):
            estimate = (investment_objective(theta + eps, horizon)[0]
                        - investment_objective(theta - eps, horizon)[0]) / (2 * eps)
            self.assertAlmostEqual(estimate, investment_objective(theta, horizon)[1], places=8)

    def test_same_initial_policy(self):
        self.assertEqual(GreedyLearner(0).action(), GreedyLearner(1).action())

    def test_frozen_does_not_adapt(self):
        self.assertEqual(reversal_lifetime(0)["total"], 0)

    def test_plastic_learner_changes_next_action(self):
        output = reversal_lifetime(1)
        self.assertEqual(output["actions"], [0] + [1] * 9)
        self.assertEqual(output["total"], 9)

    def test_small_step_response_delay(self):
        output = reversal_lifetime(0.1)
        self.assertEqual(output["actions"], [0] * 7 + [1] * 3)

    def test_tie_breaking_changes_delay(self):
        self.assertEqual(reversal_lifetime(0.5, 3)["actions"], [0, 0, 1])

    def test_learners_do_not_share_mutable_state(self):
        a, b = GreedyLearner(1), GreedyLearner(1)
        a.observe(0, 0)
        self.assertEqual(b.q, [1, 0.5])

    def test_fixed_stream_not_learner_policy(self):
        passive = fixed_stream_fit(0.1, (1,) * 10)
        actual = reversal_lifetime(0.1, 10)
        self.assertGreater(passive["final_q"][1], actual["final_q"][1])

    def test_observation_value_is_a_mixture_not_history_value(self):
        output = history_values()
        self.assertEqual(output["observation_mixture_for_action_zero"], 0.5)
        self.assertEqual(set(output["frozen_action_zero"].values()), {0.0, 1.0})

    def test_memory_can_affect_return(self):
        output = history_values()
        self.assertGreater(output["remembering_mean"], output["memoryless_mean"])

    def test_oracle_has_linear_advantage_over_any_causal_rule(self):
        for rule in (lambda h: 0.0, lambda h: 0.8, lambda h: h[-1] if h else 1):
            self.assertAlmostEqual(unpredictable_comparator(5, rule)["dynamic_regret"], 2.5)

    def test_one_step_is_known_value(self):
        records = [(a, float(a), (0.75, 0.25), (0, 1)) for a in (0, 0, 0, 1)]
        self.assertEqual(deviation_estimate(records), 0.75)

    def test_multistep_is_exact_expectation(self):
        for horizon in (1, 2, 3):
            result = exact_deviation_check(horizon)
            self.assertAlmostEqual(result["IS_expectation"], result["known_difference"])

    def test_identity_deviation_zero(self):
        self.assertEqual(segment_deviation([(1, 7, (0.5, 0.5), (0.5, 0.5))]), 0)

    def test_negative_deviation_not_clipped(self):
        self.assertEqual(segment_deviation([(1, 1, (0.5, 0.5), (1, 0))]), -1)

    def test_unsupported_target_rejected(self):
        with self.assertRaises(ValueError):
            segment_deviation([(0, 1, (1, 0), (0, 1))])

    def test_impossible_logged_action_rejected(self):
        with self.assertRaises(ValueError):
            segment_deviation([(1, 1, (1, 0), (1, 0))])

    def test_incomplete_window_rejected(self):
        with self.assertRaises(ValueError):
            deviation_estimate([(0, 1, (0.5, 0.5), (0.5, 0.5))], 2)

    def test_bad_probabilities_rejected(self):
        with self.assertRaises(ValueError):
            segment_deviation([(0, 1, (0.5, 0.6), (0.5, 0.5))])

    def test_irreversible_world_does_not_reset(self):
        risky = trap_lifetime(True, 10)
        self.assertEqual(risky["rewards"], [4] + [0] * 9)
        self.assertEqual(risky["states"], ["healthy"] + ["destroyed"] * 9)

    def test_low_local_gap_does_not_imply_no_lifetime_harm(self):
        result = trap_diagnostics(100, 6)
        self.assertEqual(result["from_initial_world_gap"], 96)
        self.assertAlmostEqual(result["mean_local_deviation_gap"], 0.02)

    def test_future_horizon_not_truncated_to_record_length(self):
        self.assertEqual(trap_diagnostics(1, 6)["mean_local_deviation_gap"], 2)

    def test_invalid_horizon_and_stepsize(self):
        with self.assertRaises(ValueError):
            investment_returns(-1)
        with self.assertRaises(ValueError):
            GreedyLearner(1.1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("demo", "test"))
    args = parser.parse_args()
    if args.mode == "test":
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ContinualControlTests))
        raise SystemExit(0 if result.wasSuccessful() else 1)
    print(json.dumps(demo(), ensure_ascii=False, indent=2, allow_nan=False))
