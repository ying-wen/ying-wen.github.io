"""Three-arm exploration mechanisms; exact teaching calculations, Python stdlib only.

Download and run from any directory:
  python3 bandit_exploration_walkthrough.py --test
  python3 bandit_exploration_walkthrough.py --json

The stationary Bernoulli means are reader-only reference values. The cold-start
trace consumes only the chosen arm's supplied reward prefix; it is one possible
history, not a performance experiment. A separate six-outcome enumeration checks
the gradient estimator against the derivative of expected reward.
"""
import argparse
import json
import math
import unittest

ARMS = ("A", "B", "C")
MEANS = (0.2, 0.5, 0.8)


def greedy(values):
    """Lowest-index tie rule, fixed throughout this walkthrough."""
    return max(range(len(values)), key=lambda a: values[a])


def epsilon_policy(values, epsilon=0.1):
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0,1]")
    chosen = greedy(values)
    return [epsilon / len(values) + (1 - epsilon) * (a == chosen)
            for a in range(len(values))]


def selected_update(values, counts, action, reward, alpha=None):
    """Return new copies; only the chosen arm receives an observation."""
    q, n = list(values), list(counts)
    n[action] += 1
    rate = 1 / n[action] if alpha is None else alpha
    if not 0 < rate <= 1:
        raise ValueError("step size must be in (0,1]")
    q[action] += rate * (reward - q[action])
    return q, n


def ucb_snapshot(values, counts, t, c=math.sqrt(2)):
    if t < 1 or c <= 0 or any(n < 0 for n in counts):
        raise ValueError("positive time and c, nonnegative counts required")
    unseen = [a for a, n in enumerate(counts) if n == 0]
    # None represents an untried arm in portable JSON; never divide by zero.
    bonus = [None if n == 0 else c * math.sqrt(math.log(t) / n)
             for n in counts]
    score = [None if b is None else q + b for q, b in zip(values, bonus)]
    action = unseen[0] if unseen else greedy(score)
    return {"t": t, "values": list(values), "counts": list(counts),
            "bonus": bonus, "score": score, "untried": unseen,
            "action": action, "epsilon_policy": epsilon_policy(values)}


def softmax(preferences):
    shift = max(preferences)
    weights = [math.exp(h - shift) for h in preferences]
    total = sum(weights)
    return [w / total for w in weights]


def preference_update(preferences, action, reward, baseline, alpha=0.3):
    probabilities = softmax(preferences)
    advantage = reward - baseline
    delta = [alpha * advantage * ((a == action) - p)
             for a, p in enumerate(probabilities)]
    after = [h + d for h, d in zip(preferences, delta)]
    return {"before": list(preferences), "policy_before": probabilities,
            "action": action, "reward": reward, "baseline_before": baseline,
            "advantage": advantage, "delta": delta, "after": after,
            "policy_after": softmax(after)}


def cold_start_trace(initial, steps=6):
    # Every listed binary prefix has positive probability under the task means.
    # A is not rewarded during this prefix; the tapes are not visible to choice.
    tapes = ((0, 0, 0, 0, 0, 0), (1, 0, 1, 0, 1, 0), (1, 1, 0, 1, 1, 0))
    q, counts, rows = [initial] * 3, [0] * 3, []
    for t in range(1, steps + 1):
        action = greedy(q)
        reward = tapes[action][counts[action]]
        before = list(q)
        q, counts = selected_update(q, counts, action, reward, alpha=0.5)
        rows.append({"t": t, "action": action, "reward": reward,
                     "before": before, "after": list(q), "counts": list(counts)})
    return {"initial": initial, "alpha": 0.5, "rows": rows,
            "final_values": q, "final_counts": counts}


def expected_reward(preferences, means=MEANS):
    return sum(p * q for p, q in zip(softmax(preferences), means))


def enumerate_gradient(preferences, baseline, include_current=False, past_count=4):
    """Enumerate every (arm, Bernoulli reward); no sampling or training."""
    p = softmax(preferences)
    outcomes, expectation = [], [0.0] * 3
    for action, mean in enumerate(MEANS):
        for reward, reward_probability in ((0, 1 - mean), (1, mean)):
            probability = p[action] * reward_probability
            used_baseline = ((past_count * baseline + reward) / (past_count + 1)
                             if include_current else baseline)
            direction = [(reward - used_baseline) * ((a == action) - p[a])
                         for a in range(3)]
            expectation = [x + probability * d for x, d in zip(expectation, direction)]
            outcomes.append({"action": action, "reward": reward,
                             "probability": probability, "baseline": used_baseline,
                             "direction": direction})
    return {"outcomes": outcomes, "expectation": expectation}


def walkthrough_data():
    # Thirteen observations: A has six successes in ten visits, B one in two,
    # C zero in one. UCB and epsilon-greedy read identical evidence here.
    history = [[1, 0, 1, 0, 1, 1, 0, 1, 0, 1], [1, 0], [0]]
    counts = [len(x) for x in history]
    q = [sum(x) / len(x) for x in history]
    before = ucb_snapshot(q, counts, 14)
    action, reward = before["action"], 1
    after_q, after_n = selected_update(q, counts, action, reward)
    after = ucb_snapshot(after_q, after_n, 15)
    h, baseline, trace = [0.0] * 3, 0.0, []
    for t, (action, reward) in enumerate(((2, 1), (2, 0)), 1):
        row = preference_update(h, action, reward, baseline)
        row["t"] = t
        baseline += (reward - baseline) / t  # Update after all preferences.
        row["baseline_after"] = baseline
        trace.append(row)
        h = row["after"]
    probe_h = [0.2, -0.1, 0.4]
    p = softmax(probe_h)
    j = expected_reward(probe_h)
    analytic = [pa * (qa - j) for pa, qa in zip(p, MEANS)]
    prior = enumerate_gradient(probe_h, 0.5)
    current = enumerate_gradient(probe_h, 0.5, include_current=True, past_count=4)
    return {"task": {"arms": ARMS, "reader_only_means": MEANS,
                     "reward": "stationary independent Bernoulli per selected arm",
                     "tie_rule": "lowest index", "evidence_kind": "exact mechanism example"},
            "cold_start": [cold_start_trace(0), cold_start_trace(1.5)],
            "selection": {"observed_rewards_by_arm": history, "before": before,
                          "selected_reward": 1, "after": after},
            "gradient_trace": trace,
            "gradient_check": {"preferences": probe_h, "baseline": 0.5,
                               "policy": p, "expected_reward": j,
                               "analytic_direction": analytic, "prior_baseline": prior,
                               "current_baseline": current, "current_scaling": 4 / 5}}


class WalkthroughTests(unittest.TestCase):
    def assertVectorNear(self, x, y, tolerance=1e-12):
        self.assertEqual(len(x), len(y))
        for a, b in zip(x, y):
            self.assertAlmostEqual(a, b, delta=tolerance)

    def test_cold_start_feedback_and_optimism_decay(self):
        plain, optimistic = walkthrough_data()["cold_start"]
        self.assertEqual([r["action"] for r in plain["rows"]], [0] * 6)
        self.assertEqual([r["action"] for r in optimistic["rows"]], [0, 1, 2, 1, 2, 2])
        self.assertVectorNear(optimistic["final_values"], [0.75, 0.625, 0.5625])
        for row in optimistic["rows"]:
            for a in range(3):
                if a != row["action"]:
                    self.assertEqual(row["before"][a], row["after"][a])
        # Independent closed expansion: C received 1,1,0, with alpha=1/2.
        self.assertAlmostEqual(optimistic["final_values"][2], 1.5 / 8 + 1 / 8 + 1 / 4)

    def test_epsilon_exact_probability_and_unseen_ucb(self):
        self.assertVectorNear(epsilon_policy([0.6, 0.5, 0]), [14 / 15, 1 / 30, 1 / 30])
        self.assertVectorNear(epsilon_policy([0, 0, 0], 1), [1 / 3] * 3)
        q, counts, choices = [0.0] * 3, [0] * 3, []
        for t in range(1, 4):
            snapshot = ucb_snapshot(q, counts, t)
            a = snapshot["action"]
            choices.append(a)
            q, counts = selected_update(q, counts, a, 0)
        self.assertEqual(choices, [0, 1, 2])

    def test_ucb_observation_changes_score_and_next_action(self):
        selection = walkthrough_data()["selection"]
        before, after = selection["before"], selection["after"]
        self.assertEqual((before["action"], after["action"]), (2, 1))
        self.assertEqual(after["counts"], [10, 2, 2])
        self.assertVectorNear(after["values"], [0.6, 0.5, 0.5])
        self.assertEqual(before["epsilon_policy"], after["epsilon_policy"])
        self.assertGreater(before["score"][2], before["score"][1])
        self.assertEqual(after["score"][1], after["score"][2])
        self.assertGreater(after["bonus"][0], before["bonus"][0])
        self.assertLess(after["bonus"][2], before["bonus"][2])

    def test_preference_trace_uses_old_policy_and_baseline(self):
        first, second = walkthrough_data()["gradient_trace"]
        self.assertVectorNear(first["after"], [-0.1, -0.1, 0.2])
        self.assertEqual(first["baseline_before"], 0)
        self.assertEqual(second["baseline_before"], 1)
        self.assertGreater(first["policy_after"][2], first["policy_before"][2])
        self.assertLess(second["policy_after"][2], second["policy_before"][2])
        self.assertAlmostEqual(sum(second["delta"]), 0)
        shifted = preference_update([5.0] * 3, 2, 8, 7)
        self.assertVectorNear(shifted["policy_after"], first["policy_after"])

    def test_enumeration_matches_independent_finite_difference(self):
        h = [0.2, -0.1, 0.4]
        dh = 1e-5
        finite_difference = []
        for a in range(3):
            left, right = list(h), list(h)
            left[a] -= dh
            right[a] += dh
            finite_difference.append((expected_reward(right) - expected_reward(left)) / (2 * dh))
        for baseline in (-10, 0, 0.5, 10):
            result = enumerate_gradient(h, baseline)
            self.assertAlmostEqual(sum(x["probability"] for x in result["outcomes"]), 1)
            self.assertVectorNear(result["expectation"], finite_difference, 1e-10)

    def test_current_reward_baseline_scales_direction(self):
        prior = enumerate_gradient([0.2, -0.1, 0.4], 0.5)["expectation"]
        for n in (0, 1, 4, 99):
            current = enumerate_gradient([0.2, -0.1, 0.4], 0.5, True, n)["expectation"]
            self.assertVectorNear(current, [n / (n + 1) * d for d in prior])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(WalkthroughTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            raise SystemExit(1)
    if args.json or not args.test:
        print(json.dumps(walkthrough_data(), ensure_ascii=False, indent=2, allow_nan=False))
