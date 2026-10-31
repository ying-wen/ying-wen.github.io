"""Finite, closed-loop on-policy Monte Carlo control. Standard library only.

Run anywhere: python3 mc-control-walkthrough.py [--test]
S0 -> S1 -> true termination; rewards (2,0)/(4,0), gamma=1/2.
The environment never reveals a value label to the learner. A complete
episode uses one frozen policy version. Given uniform numbers select its
actions by inverse CDF, then observed returns update first-visit Q averages.
An epsilon-greedy improvement follows; ties choose B consistently.

The assigned three episodes explain timing; exhaustive adaptive branches
explain what else could happen. Fraction arithmetic and a separate Bellman
oracle distinguish sample Q from exact current-policy q. There is no random
training. Sutton & Barto (2020), Sections 5.3, 5.4, 5.7 inform the mechanism;
the task, enumeration and illustrations are original.
"""
import argparse
from fractions import Fraction
from itertools import product
import json
import unittest

F = Fraction
REWARDS = ((F(2), F(0)), (F(4), F(0)))
GAMMA = F(1, 2)
EPSILON = F(1, 5)
INITIAL_POLICY = ((F(1, 2), F(1, 2)), (F(1, 2), F(1, 2)))
UNIFORMS = ((F(2, 5), F(4, 5)), (F(19, 20), F(1, 20)), (F(1, 20), F(1, 2)))


def rational(value):
    return value if isinstance(value, Fraction) else F(str(value))


def policy_pair(policy):
    if len(policy) != 2 or any(len(row) != 2 for row in policy):
        raise ValueError("Two states and two actions are required")
    result = tuple(tuple(rational(p) for p in row) for row in policy)
    if any(any(p < 0 for p in row) or sum(row) != 1 for row in result):
        raise ValueError("Each action distribution must sum to one")
    return result


def epsilon_greedy(q, epsilon=EPSILON):
    epsilon = rational(epsilon)
    if not 0 < epsilon <= 1:
        raise ValueError("Control requires 0 < epsilon <= 1")
    # Choosing B on exact ties exposes what unvisited zero entries can do.
    return tuple(tuple(epsilon / 2 + (1 - epsilon if a == (0 if row[0] > row[1] else 1) else 0)
                       for a in range(2)) for row in q)


def environment_step(state, action):
    if state not in (0, 1) or action not in (0, 1):
        raise ValueError("Illegal state/action")
    return state + 1, REWARDS[state][action], state == 1


def episode_from_actions(policy, actions):
    """Construct the observations of one possible complete episode."""
    policy = policy_pair(policy)
    if len(actions) != 2 or any(a not in (0, 1) for a in actions):
        raise ValueError("Exactly two legal actions are needed")
    state, rewards, probability, logged = 0, [], F(1), []
    for a in actions:
        sampling_probability = policy[state][a]
        if sampling_probability == 0:
            raise ValueError("The specified episode is outside policy support")
        logged.append(sampling_probability)
        probability *= sampling_probability
        state, reward, terminated = environment_step(state, a)
        rewards.append(reward)
    if not terminated:
        raise ValueError("MC requires true termination")
    returns, g = [F(0), F(0)], F(0)
    for t in (1, 0):
        g = rewards[t] + GAMMA * g
        returns[t] = g
    return {"path": "".join("AB"[a] for a in actions), "actions": list(actions),
            "rewards": rewards, "returns": returns, "probability": probability,
            "policy": policy, "logged_probabilities": logged}


def sample_episode(policy, uniforms):
    """Actual action selection, with deterministic inputs to the sampler."""
    policy = policy_pair(policy)
    if len(uniforms) != 2 or any(not 0 <= rational(u) < 1 for u in uniforms):
        raise ValueError("Provide two uniform numbers in [0,1)")
    actions = [0 if rational(u) < policy[t][0] else 1 for t, u in enumerate(uniforms)]
    row = episode_from_actions(policy, actions)
    row["uniforms"] = tuple(rational(u) for u in uniforms)
    return row


def update(q, counts, episode, epsilon=EPSILON):
    """Only observed returns enter Q. The small model is not an input."""
    new_q = [list(row) for row in q]
    new_n = [list(row) for row in counts]
    # Each state occurs once here. This is also its first state-action visit.
    for t, a in enumerate(episode["actions"]):
        new_n[t][a] += 1
        new_q[t][a] += (episode["returns"][t] - new_q[t][a]) / new_n[t][a]
    new_q, new_n = tuple(map(tuple, new_q)), tuple(map(tuple, new_n))
    return new_q, new_n, epsilon_greedy(new_q, epsilon)


def exact_value(policy):
    """Reader-only oracle: recurse over states, not sampled returns."""
    policy = policy_pair(policy)
    values = [F(0), F(0), F(0)]
    q = [[F(0), F(0)], [F(0), F(0)]]
    for s in (1, 0):
        q[s] = [REWARDS[s][a] + GAMMA * values[s + 1] for a in range(2)]
        values[s] = sum(policy[s][a] * q[s][a] for a in range(2))
    return {"values": values[:2], "q": q}


def adaptive_branches(episodes=2, epsilon=EPSILON):
    """Enumerate policies that change after each episode, not iid paths."""
    if not isinstance(episodes, int) or not 1 <= episodes <= 3:
        raise ValueError("Enumerate one to three episodes")
    nodes = [{"paths": [], "probability": F(1), "q": ((F(0), F(0)),) * 2,
              "counts": ((0, 0),) * 2, "policy": INITIAL_POLICY}]
    for _ in range(episodes):
        children = []
        for node in nodes:
            for actions in product(range(2), repeat=2):
                row = episode_from_actions(node["policy"], actions)
                q, counts, policy = update(node["q"], node["counts"], row, epsilon)
                children.append({"paths": node["paths"] + [row["path"]],
                    "probability": node["probability"] * row["probability"],
                    "q": q, "counts": counts, "policy": policy})
        nodes = children
    for node in nodes:
        node["true_current_value"] = exact_value(node["policy"])["values"][0]
    return {"episodes": episodes, "branches": nodes,
            "probability_sum": sum(node["probability"] for node in nodes),
            "expected_current_value": sum(node["probability"] * node["true_current_value"] for node in nodes),
            "below_initial_probability": sum(node["probability"] for node in nodes if node["true_current_value"] < 2)}


def frozen_regime_mixture():
    """A separate estimator diagnostic with two prechosen frozen regimes.

    Condition the first action on S0,A. Then only S1 is sampled. This is
    conditional q evaluation, not the adaptive controller above. Enumerate
    one independent old-policy and one independent new-policy return.
    """
    old, current = F(1, 2), F(9, 10)
    batches = []
    for a, b in product(range(2), repeat=2):
        probability = (old if a == 0 else 1 - old) * (current if b == 0 else 1 - current)
        returns = (2 + GAMMA * REWARDS[1][a], 2 + GAMMA * REWARDS[1][b])
        batches.append({"actions_at_S1": [a, b], "probability": probability,
                        "returns": returns, "running_average": sum(returns) / 2})
    return {"old_A_probability": old, "current_A_probability": current,
            "old_q": 2 + 2 * old, "current_q": 2 + 2 * current,
            "batches": batches,
            "expected_running_average": sum(row["probability"] * row["running_average"] for row in batches)}


def suffix_weights(p=F(9, 10), lengths=(1, 2, 5, 10, 20, 50), budget=100):
    """A separate H-step suffix: zero rewards then terminal unit reward.

    Target chooses A at every remaining decision; behavior independently
    chooses A with p. gamma=1. Complete IS weight is 0 after any mismatch,
    otherwise p**(-H). Match/no-match enumeration is a Bernoulli oracle.
    """
    p = rational(p)
    if not 0 < p <= 1 or any(not isinstance(h, int) or h < 1 for h in lengths):
        raise ValueError("Positive support and positive integer lengths required")
    return [{"remaining_decisions": h, "match_probability": p ** h,
             "nonzero_weight": p ** (-h), "target_mean": F(1),
             "weight_variance": p ** (-h) - 1,
             "expected_matches": budget * p ** h,
             "zero_matches_probability": (1 - p ** h) ** budget}
            for h in lengths]


def walkthrough():
    q, counts, policy = ((F(0), F(0)),) * 2, ((0, 0),) * 2, INITIAL_POLICY
    rows = []
    for i, uniforms in enumerate(UNIFORMS):
        row = sample_episode(policy, uniforms)
        row.update({"episode": i + 1, "q_before": q, "counts_before": counts,
                    "exact_before": exact_value(policy)})
        q, counts, policy = update(q, counts, row)
        row.update({"q_after": q, "counts_after": counts, "policy_after": policy,
                    "exact_after": exact_value(policy)})
        rows.append(row)
    return {"kind": "exact_finite_control", "gamma": GAMMA, "epsilon": EPSILON,
            "rewards": REWARDS, "initial_policy": INITIAL_POLICY,
            "initial_value": exact_value(INITIAL_POLICY)["values"][0],
            "assigned_episodes": rows, "adaptive_enumerations": [adaptive_branches(1), adaptive_branches(2)],
            "frozen_regime_mixture": frozen_regime_mixture(),
            "coverage_cost": {"optimal_soft_policy": ((F(9, 10), F(1, 10)),) * 2,
                "soft_value": F(18, 5), "greedy_value": F(4), "value_cost": F(2, 5),
                "nonpreferred_action_probability": F(1, 10), "mean_wait_episodes": F(10),
                "no_B_in_10_probability": F(9, 10) ** 10},
            "long_suffix": {"behavior_A_probability": F(9, 10), "budget": 100,
                "rows": suffix_weights(), "uniform_behavior_H20": suffix_weights(F(1, 2), (20,))[0]}}


def json_numbers(value):
    if isinstance(value, Fraction):
        return float(value)
    if isinstance(value, dict):
        return {key: json_numbers(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_numbers(item) for item in value]
    return value


class WalkthroughTests(unittest.TestCase):
    def test_actual_closed_loop(self):
        rows = walkthrough()["assigned_episodes"]
        self.assertEqual([row["path"] for row in rows], ["AB", "BA", "AA"])
        self.assertEqual([row["probability"] for row in rows], [F(1, 4), F(1, 100), F(9, 100)])
        self.assertEqual(rows[-1]["q_after"], ((3, 2), (4, 0)))
        self.assertEqual(rows[-1]["counts_after"], ((2, 1), (2, 1)))
        self.assertEqual(rows[-1]["exact_after"]["q"][0][0], F(19, 5))

    def test_branches_include_expected_failure(self):
        first = adaptive_branches(1)
        self.assertEqual(first["probability_sum"], 1)
        self.assertEqual(first["expected_current_value"], 2)
        self.assertEqual(first["below_initial_probability"], F(1, 4))
        self.assertEqual(next(n for n in first["branches"] if n["paths"] == ["BB"])["true_current_value"], F(2, 5))
        self.assertEqual(len(adaptive_branches(2)["branches"]), 16)
        self.assertEqual(adaptive_branches(2)["probability_sum"], 1)

    def test_separate_frozen_policy_diagnostic(self):
        mixed = frozen_regime_mixture()
        self.assertEqual((mixed["old_q"], mixed["current_q"], mixed["expected_running_average"]), (3, F(19, 5), F(17, 5)))
        self.assertNotEqual(mixed["expected_running_average"], mixed["current_q"])

    def test_exact_values_and_coverage_cost(self):
        self.assertEqual(exact_value(INITIAL_POLICY)["q"], [[3, 1], [4, 0]])
        self.assertEqual(exact_value(((F(9, 10), F(1, 10)),) * 2)["values"][0], F(18, 5))
        self.assertEqual(epsilon_greedy(((4, 0), (4, 0)), 1), INITIAL_POLICY)
        self.assertEqual(exact_value(((1, 0),) * 2)["values"][0], 4)

    def test_suffix_matches_enumeration(self):
        for h in (1, 2, 5):
            weights = []
            for actions in product(range(2), repeat=h):
                probability = F(1)
                for a in actions:
                    probability *= F(9, 10) if a == 0 else F(1, 10)
                weights.append((probability, F(10, 9) ** h if all(a == 0 for a in actions) else F(0)))
            row = suffix_weights(lengths=(h,))[0]
            self.assertEqual(sum(p * w for p, w in weights), 1)
            self.assertEqual(sum(p * (w - 1) ** 2 for p, w in weights), row["weight_variance"])

    def test_boundary_inputs_and_inverse_cdf(self):
        self.assertEqual(sample_episode(INITIAL_POLICY, (F(1, 2), 0))["path"], "BA")
        for epsilon in (0, -1, 2):
            with self.assertRaises(ValueError):
                epsilon_greedy(((0, 0), (0, 0)), epsilon)
        for uniforms in ((0, 1), (-1, 0), (0,)):
            with self.assertRaises(ValueError):
                sample_episode(INITIAL_POLICY, uniforms)
        with self.assertRaises(ValueError):
            episode_from_actions(((0, 1), (1, 0)), (0, 0))
        with self.assertRaises(ValueError):
            suffix_weights(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    if args.test:
        unittest.main(argv=["mc-control-walkthrough.py"], verbosity=2)
    else:
        print(json.dumps(json_numbers(walkthrough()), ensure_ascii=False, indent=2))
