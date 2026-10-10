"""Two-step off-policy Monte Carlo calculation, using only Python's standard library.

Run anywhere: python3 /path/to/monte-carlo-walkthrough.py
Check:        python3 /path/to/monte-carlo-walkthrough.py --test

Every episode starts at S0, moves to S1, then truly terminates. Actions A/B
give rewards (2, 0) at S0 and (4, 0) at S1. Actions do not change the next
state. gamma=1/2, b(A|S0)=b(A|S1)=1/2, pi(A|S0)=pi(A|S1)=3/4.
There is one visit to each state per episode. The evaluated policy pi stays
fixed. No random sampling, environment training, or external files are used.

The four trajectories and all 16 ordered two-episode batches are enumerated
exactly with Fraction. They distinguish a sample target, its expectation,
and the expectation of a normalized finite-batch estimator. The assigned
AA/BB batch is a possible observation, not a comparison of method performance.

Definitions checked against Sutton & Barto, second edition, Sections 5.5
and 5.9, Equations (5.3)-(5.6), (5.11)-(5.14). The task is original.
"""
import argparse
from fractions import Fraction
from itertools import product
import json
import unittest


F = Fraction
BEHAVIOR = ((F(1, 2), F(1, 2)), (F(1, 2), F(1, 2)))
TARGET = ((F(3, 4), F(1, 4)), (F(3, 4), F(1, 4)))
REWARDS = ((F(2), F(0)), (F(4), F(0)))


def rational(value):
    """Decimal input keeps its intended decimal value, e.g. 0.5 -> 1/2."""
    return value if isinstance(value, Fraction) else F(str(value))


def policy_pair(policy):
    if len(policy) != 2 or any(len(row) != 2 for row in policy):
        raise ValueError("A two-state, two-action policy is required")
    result = tuple(tuple(rational(x) for x in row) for row in policy)
    if any(any(x < 0 for x in row) or sum(row) != 1 for row in result):
        raise ValueError("Action probabilities must be nonnegative and sum to one")
    return result


def trajectory(actions, logged_behavior, target=TARGET, gamma=F(1, 2)):
    """Use each action's recorded sampling probability as the denominator.

    A changed current behavior policy is deliberately absent from this API.
    Rewards are determined by the actions; pi is held fixed for this call.
    For state value at S0, the first action IS ratio must be included.
    """
    if len(actions) != 2 or any(a not in (0, 1) for a in actions):
        raise ValueError("An episode must contain exactly two legal actions")
    if len(logged_behavior) != 2:
        raise ValueError("Record one sampling probability for each action")
    pi, gamma = policy_pair(target), rational(gamma)
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must lie in [0, 1]")
    denominators = tuple(rational(p) for p in logged_behavior)
    if any(not 0 < p <= 1 for p in denominators):
        raise ValueError("An observed action must have positive sampling probability")
    rewards = tuple(REWARDS[t][a] for t, a in enumerate(actions))
    ratios = tuple(pi[t][a] / denominators[t] for t, a in enumerate(actions))
    rho = ratios[0] * ratios[1]
    returned = rewards[0] + gamma * rewards[1]
    # Only the second reward needs both action ratios. Later actions cannot
    # change an already observed R1; its prefix ratio contains A0 alone.
    contributions = (ratios[0] * rewards[0], gamma * rho * rewards[1])
    return {"path": "".join("AB"[a] for a in actions),
            "actions": list(actions), "rewards": list(rewards),
            "logged_behavior": list(denominators), "ratios": list(ratios),
            "weight": rho, "return": returned,
            "ordinary_target": rho * returned,
            "per_decision_contributions": list(contributions),
            "per_decision_target": sum(contributions)}


def enumerate_trajectories(behavior=BEHAVIOR, target=TARGET, gamma=F(1, 2)):
    """The model enumerates observations; an estimator never queries it."""
    b, pi = policy_pair(behavior), policy_pair(target)
    if any(pi[t][a] > 0 and b[t][a] == 0 for t in range(2) for a in range(2)):
        raise ValueError("Behavior must cover the target's actions")
    rows = []
    for actions in product(range(2), repeat=2):
        probability = b[0][actions[0]] * b[1][actions[1]]
        if probability == 0:
            continue  # Unobservable paths cannot supply a sampled return.
        row = trajectory(actions, [b[t][a] for t, a in enumerate(actions)], pi, gamma)
        row["behavior_probability"] = probability
        row["target_probability"] = pi[0][actions[0]] * pi[1][actions[1]]
        rows.append(row)
    return rows


def estimate_batch(rows):
    if not rows:
        raise ValueError("A batch must contain at least one complete episode")
    numerator = sum(row["ordinary_target"] for row in rows)
    denominator = sum(row["weight"] for row in rows)
    return {"paths": [row["path"] for row in rows], "episodes": len(rows),
            "weighted_return_sum": numerator, "weight_sum": denominator,
            "ordinary": numerator / len(rows),
            "weighted": numerator / denominator if denominator else None,
            "per_decision": sum(row["per_decision_target"] for row in rows) / len(rows)}


def moments(rows, key):
    """A probability-weighted population calculation, not a sample average."""
    mean = sum(row["behavior_probability"] * row[key] for row in rows)
    variance = sum(row["behavior_probability"] * (row[key] - mean) ** 2 for row in rows)
    return {"mean": mean, "variance": variance}


def batch_expectations(rows, n=2):
    if not isinstance(n, int) or not 1 <= n <= 6:
        raise ValueError("Enumerate between one and six independent episodes")
    means = {key: F(0) for key in ("ordinary", "weighted", "per_decision")}
    undefined_probability = F(0)
    for batch in product(rows, repeat=n):
        # Episodes are independent here, so the batch probability is a product.
        probability = F(1)
        for row in batch:
            probability *= row["behavior_probability"]
        estimates = estimate_batch(batch)
        for key in means:
            if estimates[key] is not None:
                means[key] += probability * estimates[key]
        if estimates["weighted"] is None:
            undefined_probability += probability
    # Do not silently assign a value to an undefined weighted estimator.
    if undefined_probability:
        means["weighted"] = None
    return {"episodes": n, "ordered_batches": len(rows) ** n,
            "means": means, "weighted_undefined_probability": undefined_probability}


def walkthrough():
    rows = enumerate_trajectories()
    true_value = sum(TARGET[0][a] * REWARDS[0][a] for a in range(2))
    true_value += F(1, 2) * sum(TARGET[1][a] * REWARDS[1][a] for a in range(2))
    # Only the *current* S1 behavior changes, after the old data were collected.
    new_behavior = (BEHAVIOR[0], (F(1, 4), F(3, 4)))
    incorrect_rows = []
    for old in rows:
        actions = old["actions"]
        incorrect = trajectory(actions, [new_behavior[t][a] for t, a in enumerate(actions)])
        # Sampling still happened under the OLD distribution. Replacing this
        # probability as well would conceal the denominator error.
        incorrect["behavior_probability"] = old["behavior_probability"]
        incorrect_rows.append(incorrect)
    batch2 = batch_expectations(rows, 2)
    wis2 = batch2["means"]["weighted"]
    return {"kind": "exact_finite_enumeration", "gamma": F(1, 2),
            "behavior": BEHAVIOR, "target": TARGET, "rewards": REWARDS,
            "true_value": true_value, "rows": rows,
            "population": {"ordinary": moments(rows, "ordinary_target"),
                           "per_decision": moments(rows, "per_decision_target")},
            "assigned_batch": estimate_batch([rows[0], rows[3]]),
            "batch_expectations": [batch_expectations(rows, 1), batch2],
            "two_episode_wis_exact": str(wis2),
            "two_episode_wis_bias": wis2 - true_value,
            "behavior_change": {"new_behavior": new_behavior,
                "old_AA": rows[0], "wrong_AA": incorrect_rows[0],
                "wrong_old_data_ordinary_mean": moments(incorrect_rows, "ordinary_target")["mean"],
                "wrong_old_data_per_decision_mean": moments(incorrect_rows, "per_decision_target")["mean"],
                "new_data_correct_ordinary_mean": moments(enumerate_trajectories(new_behavior), "ordinary_target")["mean"]}}


def json_numbers(value):
    if isinstance(value, Fraction):
        return float(value)
    if isinstance(value, dict):
        return {key: json_numbers(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_numbers(item) for item in value]
    return value


class WalkthroughTests(unittest.TestCase):
    def test_hand_calculation(self):
        data = walkthrough()
        self.assertEqual(data["true_value"], 3)
        self.assertEqual([row["ordinary_target"] for row in data["rows"]], [9, F(3, 2), F(3, 2), 0])
        self.assertEqual([row["per_decision_target"] for row in data["rows"]], [F(15, 2), 3, F(3, 2), 0])
        batch = data["assigned_batch"]
        self.assertEqual((batch["ordinary"], batch["weighted"], batch["per_decision"]), (F(9, 2), F(18, 5), F(15, 4)))

    def test_population_and_batches(self):
        rows = enumerate_trajectories()
        self.assertEqual(moments(rows, "ordinary_target"), {"mean": 3, "variance": F(99, 8)})
        self.assertEqual(moments(rows, "per_decision_target"), {"mean": 3, "variance": F(63, 8)})
        for n in (1, 2):
            means = batch_expectations(rows, n)["means"]
            self.assertEqual(means["ordinary"], 3)
            self.assertEqual(means["per_decision"], 3)
        self.assertEqual(batch_expectations(rows, 1)["means"]["weighted"], 2)

    def test_gamma_zero_removes_second_reward(self):
        rows = enumerate_trajectories(gamma=0)
        self.assertEqual([row["per_decision_target"] for row in rows], [3, 3, 0, 0])
        self.assertEqual(moments(rows, "per_decision_target")["mean"], F(3, 2))

    def test_on_policy(self):
        rows = enumerate_trajectories(target=BEHAVIOR)
        self.assertTrue(all(row["weight"] == 1 and row["ordinary_target"] == row["per_decision_target"] == row["return"] for row in rows))
        batch = estimate_batch([rows[0], rows[3]])
        self.assertEqual(batch["ordinary"], batch["weighted"])

    def test_zero_weight_and_missing_support(self):
        deterministic = ((1, 0), (1, 0))
        rows = enumerate_trajectories(target=deterministic)
        self.assertIsNone(estimate_batch([rows[3]])["weighted"])
        self.assertEqual(batch_expectations(rows, 2)["weighted_undefined_probability"], F(9, 16))
        with self.assertRaises(ValueError):
            enumerate_trajectories(behavior=((0, 1), (F(1, 2), F(1, 2))))

    def test_recorded_probability(self):
        change = walkthrough()["behavior_change"]
        self.assertEqual(change["old_AA"]["weight"], F(9, 4))
        self.assertEqual(change["wrong_AA"]["weight"], F(9, 2))
        self.assertEqual(change["wrong_old_data_ordinary_mean"], F(11, 2))
        self.assertEqual(change["wrong_old_data_per_decision_mean"], F(9, 2))
        self.assertEqual(change["new_data_correct_ordinary_mean"], 3)

    def test_input_boundaries(self):
        for probability in (0, -1, 2):
            with self.assertRaises(ValueError):
                trajectory((0, 0), (F(1, 2), probability))
        with self.assertRaises(ValueError):
            enumerate_trajectories(target=((F(1, 2), F(1, 3)), TARGET[1]))
        with self.assertRaises(ValueError):
            trajectory((0, 0), (F(1, 2), F(1, 2)), gamma=2)
        with self.assertRaises(ValueError):
            estimate_batch([])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", help="Run exact arithmetic and boundary checks")
    args = parser.parse_args()
    if args.test:
        unittest.main(argv=["monte-carlo-walkthrough.py"], verbosity=2)
    else:
        print(json.dumps(json_numbers(walkthrough()), ensure_ascii=False, indent=2))
