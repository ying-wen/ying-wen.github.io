"""Reward design: exact counterexamples and gradient checks (Python stdlib).

Original teaching examples, not a reproduction of a neural RL benchmark.
Usage: python3 reward_design_lab.py demo | test
Code: MIT. Tutorial: CC BY 4.0.
"""
import argparse
import math
import unittest
from collections import Counter


# BEGIN expressivity
def additive_return(events, weights):
    return sum(weights.get(event, 0.0) for event in events)


def ordered_goal(events):
    """Finite-state reward: collect key before door; pay once."""
    mode, reward = "start", 0
    for event in events:
        if mode == "start" and event == "key":
            mode = "has-key"
        elif mode == "has-key" and event == "door":
            mode, reward = "done", reward + 1
    return reward
# END expressivity


# BEGIN shaping
def discounted_sum(rewards, gamma):
    return sum(gamma ** t * r for t, r in enumerate(rewards))


def shape(rewards, potentials, gamma):
    """potentials[t] is Phi_t(s_t); retain the final boundary potential."""
    if len(potentials) != len(rewards) + 1:
        raise ValueError("One potential is required at each boundary.")
    return [r + gamma * potentials[t + 1] - potentials[t]
            for t, r in enumerate(rewards)]


def shaping_residual(rewards, potentials, gamma):
    lhs = discounted_sum(shape(rewards, potentials, gamma), gamma)
    rhs = (discounted_sum(rewards, gamma) - potentials[0]
           + gamma ** len(rewards) * potentials[-1])
    return lhs - rhs
# END shaping


# BEGIN preference
def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def preference_loss_gradient(theta, feature_difference, label):
    """label=1 prefers A. Difference is f(A)-f(B); temperature is fixed at 1."""
    if not 0.0 <= label <= 1.0:
        raise ValueError("Preference label must lie in [0, 1].")
    z = theta * feature_difference
    loss = max(z, 0.0) - label * z + math.log1p(math.exp(-abs(z)))
    gradient = (sigmoid(z) - label) * feature_difference
    return loss, gradient
# END preference


# BEGIN maxent
def maxent_loss_gradient(theta, features, empirical_mean):
    """Finite, equally weighted feasible trajectories; deterministic toy model."""
    logits = [theta * f for f in features]
    peak = max(logits)
    unnormalized = [math.exp(x - peak) for x in logits]
    total = sum(unnormalized)
    probs = [x / total for x in unnormalized]
    log_z = peak + math.log(total)
    expected_feature = sum(p * f for p, f in zip(probs, features))
    return log_z - theta * empirical_mean, expected_feature - empirical_mean
# END maxent


# BEGIN intrinsic
def intrinsic_meta_gradient(theta, eta, alpha):
    """One expected-gradient bandit update; theta is independent of eta here.

    Extrinsic rewards are (0, 1); intrinsic reward difference is eta.
    The outer objective is the new probability of action 1, not its bonus.
    """
    p = sigmoid(theta)
    sensitivity = alpha * p * (1 - p)
    next_theta = theta + sensitivity * (1 + eta)
    outer_return = sigmoid(next_theta)
    gradient = outer_return * (1 - outer_return) * sensitivity
    return outer_return, gradient
# END intrinsic


# BEGIN constraint
def constrained_mixture(budget):
    """Safe action: (reward,cost)=(1,0); risky: (3,1).

    Choose risky with probability p. E[cost]<=budget; this is NOT a
    per-trajectory guarantee. The risk limit is an expectation constraint.
    """
    if not 0 <= budget <= 1:
        raise ValueError("Budget must lie in [0, 1].")
    p = budget
    return {"p_risky": p, "expected_reward": 1 + 2 * p,
            "expected_cost": p}
# END constraint


def finite_difference(f, x, eps=1e-6):
    return (f(x + eps) - f(x - eps)) / (2 * eps)


class RewardDesignTests(unittest.TestCase):
    def test_event_counts_identical(self):
        self.assertEqual(Counter(["key", "door"]), Counter(["door", "key"]))

    def test_event_rewards_cannot_rank_order(self):
        for weights in ({"key": 2, "door": -3}, {"key": 0, "door": 5}):
            self.assertEqual(additive_return(["key", "door"], weights),
                             additive_return(["door", "key"], weights))

    def test_automaton_distinguishes_order(self):
        self.assertEqual(ordered_goal(["key", "door"]), 1)
        self.assertEqual(ordered_goal(["door", "key"]), 0)

    def test_automaton_pays_only_once(self):
        self.assertEqual(ordered_goal(["key", "door", "door"]), 1)

    def test_shaping_telescope(self):
        for gamma in (0, .5, .9, 1):
            self.assertAlmostEqual(shaping_residual([0, -1, 3], [2, 4, -3, 7], gamma), 0)

    def test_terminal_zero_preserves_gap(self):
        a, b = [0, 2], [1, 0]
        gap = discounted_sum(a, .9) - discounted_sum(b, .9)
        self.assertAlmostEqual(discounted_sum(shape(a, [4, 1, 0], .9), .9)
                               - discounted_sum(shape(b, [4, 8, 0], .9), .9), gap)

    def test_nonzero_terminal_changes_ranking(self):
        self.assertGreater(2, 1)
        self.assertLess(shape([2], [0, 0], .9)[0], shape([1], [0, 10], .9)[0])

    def test_shape_input(self):
        with self.assertRaises(ValueError):
            shape([1, 2], [0, 0], .9)

    def test_time_varying_potential(self):
        self.assertAlmostEqual(shaping_residual([0, 1], [0, 2, 0], .9), 0)

    def test_wrong_time_index_not_telescoping(self):
        # Old Phi_0(s1)=2 but new Phi_1(s1)=5; Phi_0(s0)=Phi_1(s2)=0.
        wrong = [.9 * 2 - 0, .9 * 0 - 5]
        self.assertAlmostEqual(discounted_sum(wrong, .9), -2.7)

    def test_constant_bonus_changes_variable_horizon(self):
        short, long = [1], [0, 0, .9]
        self.assertGreater(sum(short), sum(long))
        self.assertLess(sum(r + .2 for r in short), sum(r + .2 for r in long))

    def test_constant_bonus_fixed_horizon(self):
        a, b = [1, 0], [0, .5]
        self.assertAlmostEqual(sum(a) - sum(b), sum(x + .2 for x in a) - sum(x + .2 for x in b))

    def test_positive_scaling_preserves_return_order(self):
        self.assertGreater(discounted_sum([0, 6], .9), discounted_sum([3, 0], .9))

    def test_preference_gradient(self):
        for label in (0, .5, 1):
            loss = lambda t: preference_loss_gradient(t, 2.3, label)[0]
            self.assertAlmostEqual(finite_difference(loss, .7),
                                   preference_loss_gradient(.7, 2.3, label)[1], places=7)

    def test_preference_label_direction(self):
        _, grad = preference_loss_gradient(0, 2, 1)
        self.assertLess(grad, 0)
        self.assertGreater(sigmoid(-.1 * grad * 2), .5)

    def test_preference_extremes(self):
        for z in (-1000, 1000):
            loss, grad = preference_loss_gradient(z, 1, 1)
            self.assertTrue(math.isfinite(loss) and math.isfinite(grad))

    def test_preference_tie(self):
        self.assertEqual(preference_loss_gradient(0, 3, .5)[1], 0)

    def test_preference_label_validation(self):
        with self.assertRaises(ValueError):
            preference_loss_gradient(0, 1, 2)

    def test_maxent_gradient(self):
        loss = lambda t: maxent_loss_gradient(t, [-1, 0, 2], 1)[0]
        self.assertAlmostEqual(finite_difference(loss, .4),
                               maxent_loss_gradient(.4, [-1, 0, 2], 1)[1], places=7)

    def test_maxent_feature_matching(self):
        self.assertAlmostEqual(maxent_loss_gradient(0, [0, 1, 2], 1)[1], 0)

    def test_intrinsic_meta_gradient(self):
        for eta in (-3, 0, 3):
            objective = lambda e: intrinsic_meta_gradient(.4, e, .2)[0]
            self.assertAlmostEqual(finite_difference(objective, eta),
                                   intrinsic_meta_gradient(.4, eta, .2)[1], places=7)

    def test_intrinsic_no_update_no_meta_effect(self):
        self.assertEqual(intrinsic_meta_gradient(.4, 2, 0), (sigmoid(.4), 0))

    def test_proxy_changes_selected_action(self):
        true_rewards, proxy = [2, 1, -3], [2, 1, 10]
        selected = max(range(3), key=lambda i: proxy[i])
        self.assertEqual(true_rewards[selected], -3)

    def test_expected_constraint(self):
        result = constrained_mixture(.2)
        self.assertEqual(result["p_risky"], .2)
        self.assertAlmostEqual(result["expected_reward"], 1.4)

    def test_lagrangian_tie_does_not_choose_feasible_mixture(self):
        multiplier = 2
        self.assertEqual(1, 3 - multiplier)
        # All mixtures tie under penalized reward. Some violate cost<=.2.
        self.assertGreater(constrained_mixture(1)["expected_cost"], .2)

    def test_budget_validation(self):
        with self.assertRaises(ValueError):
            constrained_mixture(-.1)


def demo():
    print("Order-sensitive goal:", ordered_goal(["key", "door"]), ordered_goal(["door", "key"]))
    print("Shaping identity residual:", shaping_residual([0, -1, 3], [2, 4, -3, 7], .9))
    print("Preference loss, gradient:", preference_loss_gradient(0, 2, 1))
    print("MaxEnt NLL, gradient:", maxent_loss_gradient(.4, [-1, 0, 2], 1))
    print("Intrinsic reward outer return, meta-gradient:", intrinsic_meta_gradient(.4, 0, .2))
    print("Expected-cost constrained optimum:", constrained_mixture(.2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["demo", "test"])
    args = parser.parse_args()
    if args.command == "test":
        unittest.main(argv=[__file__])
    else:
        demo()
