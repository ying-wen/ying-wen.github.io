#!/usr/bin/env python3
"""Exact two-ply self-play/search-label lesson. Standard library only.

Run: python3 tutorials/selfplay_search_labels.py [test]
No neural network, optimizer, stochastic self-play, or training is run.
Each simulation reaches a known terminal; this is NOT full AlphaZero MCTS.
PUCT-like teaching score: Q + P sqrt(1 + total visits) / (1 + edge visits).
Initial Q=0, ties choose the first action, no root exploration noise.
"""
import json
import math
import sys
import unittest

PAYOFFS = [[1, -1], [0, 1]]  # player 1 utility; player 2 receives its negative
PRIOR = [0.8, 0.2]


def search_labels(simulations=32):
    if not isinstance(simulations, int) or simulations < 1:
        raise ValueError("positive integer simulations required")
    counts, sums = [0, 0], [0, 0]
    child_counts, child_sums = [[0, 0], [0, 0]], [[0, 0], [0, 0]]
    trace = []

    def select(n, w, prior):
        scores = [(w[a] / n[a] if n[a] else 0)
                  + prior[a] * math.sqrt(1 + sum(n)) / (1 + n[a])
                  for a in range(2)]
        return max(range(2), key=lambda a: scores[a])

    for k in range(simulations):
        a = select(counts, sums, PRIOR)
        b = select(child_counts[a], child_sums[a], [0.5, 0.5])
        z = PAYOFFS[a][b]
        counts[a] += 1
        sums[a] += z
        child_counts[a][b] += 1
        child_sums[a][b] -= z  # opponent maximizes THEIR utility, not player 1's
        trace.append(dict(simulation=k + 1, actions=[a, b], rootOutcome=z,
                          rootVisits=counts.copy()))
    policy = [n / simulations for n in counts]
    return dict(simulations=simulations, payoffs=PAYOFFS, prior=PRIOR,
                counts=counts, childCounts=child_counts,
                actionValues=[w / n if n else 0 for n, w in zip(counts, sums)],
                policy=policy, minimax=[min(row) for row in PAYOFFS], trace=trace,
                exampleGame=dict(actions=[0, 1], valueLabels=[-1, 1]),
                policyLogitGradient=[p - pi for p, pi in zip(PRIOR, policy)])


class SearchLabelTests(unittest.TestCase):
    def test_first_simulations_are_hand_checkable(self):
        self.assertEqual([x["actions"] for x in search_labels(4)["trace"]],
                         [[0, 0], [0, 1], [0, 1], [1, 0]])

    def test_search_visits_are_not_prior_or_minimax_value(self):
        d = search_labels()
        self.assertEqual(d["counts"], [6, 26])
        self.assertEqual(d["childCounts"], [[1, 5], [24, 2]])
        self.assertEqual(d["policy"], [3 / 16, 13 / 16])
        self.assertEqual(d["minimax"], [-1, 0])
        self.assertEqual(d["actionValues"], [-2 / 3, 1 / 13])

    def test_small_budget_does_not_guarantee_improvement(self):
        self.assertEqual(search_labels(1)["policy"], [1, 0])

    def test_outcome_changes_sign_with_player(self):
        self.assertEqual(search_labels()["exampleGame"]["valueLabels"], [-1, 1])

    def test_cross_entropy_logit_gradient(self):
        d, eps = search_labels(), 1e-6
        logits = [math.log(p) for p in PRIOR]
        def loss(xs):
            normalizer = math.log(sum(math.exp(x) for x in xs))
            return -sum(pi * (x - normalizer) for pi, x in zip(d["policy"], xs))
        for a in range(2):
            plus, minus = logits.copy(), logits.copy()
            plus[a] += eps
            minus[a] -= eps
            self.assertAlmostEqual((loss(plus)-loss(minus))/(2*eps),
                                   d["policyLogitGradient"][a], places=8)


if __name__ == "__main__":
    if sys.argv[1:] == ["test"]:
        unittest.main(argv=[sys.argv[0]])
    elif not sys.argv[1:]:
        print(json.dumps(search_labels(), indent=2))
    else:
        raise SystemExit("usage: selfplay_search_labels.py [test]")
