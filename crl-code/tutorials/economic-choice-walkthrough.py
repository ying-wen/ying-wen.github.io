#!/usr/bin/env python3
"""Finite lotteries: calculation is separate from human questionnaire observations.

Run: python3 economic-choice-walkthrough.py [--test]
Only Python's standard library is needed; no training or simulated participants.
Lottery specifications: Kahneman & Tversky (1979), Problems 1–4 and 11–12.
https://doi.org/10.2307/1914185
"""
import argparse
from fractions import Fraction as F
import json
import math
import unittest


LOTTERIES = {
    "p1A": [(2500, F(33, 100)), (2400, F(66, 100)), (0, F(1, 100))],
    "p1B": [(2400, F(1))],
    "p2C": [(2500, F(33, 100)), (0, F(67, 100))],
    "p2D": [(2400, F(34, 100)), (0, F(66, 100))],
    "p3A": [(4000, F(4, 5)), (0, F(1, 5))], "p3B": [(3000, F(1))],
    "p4C": [(4000, F(1, 5)), (0, F(4, 5))], "p4D": [(3000, F(1, 4)), (0, F(3, 4))],
    "p11A": [(1000, F(1, 2)), (0, F(1, 2))], "p11B": [(500, F(1))],
    "p12C": [(-1000, F(1, 2)), (0, F(1, 2))], "p12D": [(-500, F(1))],
}
WEIGHT_NODES = [(0, 0), (.2, .25), (.25, .28), (.8, .65), (1, 1)]


def expected(lottery, utility=lambda x: x):
    """Probabilities are rational; the utility can be rational or floating point."""
    if sum(p for _, p in lottery) != 1 or any(p < 0 for _, p in lottery):
        raise ValueError("Probabilities must sum to one")
    return sum(p * utility(x) for x, p in lottery)


def weight(p):
    """Constructed monotone piecewise straight lines, not fitted behavior."""
    if not 0 <= p <= 1:
        raise ValueError("p must be in [0, 1]")
    for (a, wa), (b, wb) in zip(WEIGHT_NODES, WEIGHT_NODES[1:]):
        if p <= b:
            return wa + (p - a) * (wb - wa) / (b - a)
    return 1.0


def reference_value(delta):
    """delta in thousands of pounds; a deliberately chosen mechanism function."""
    return math.sqrt(delta) if delta >= 0 else -2 * math.sqrt(-delta)


def final_distribution(lottery, gift):
    """Aggregate equal terminal amounts after adding the hypothetical gift."""
    result = {}
    for delta, p in lottery:
        result[gift + delta] = result.get(gift + delta, F(0)) + p
    return result


def calculate():
    means = {name: int(expected(lottery)) for name, lottery in LOTTERIES.items()}
    weighted = {"p3A": weight(.8), "p3B": .75, "p4C": weight(.2), "p4D": weight(.25) * .75}
    reference = {name: expected(LOTTERIES[name], lambda delta: reference_value(delta / 1000))
                 for name in ("p11A", "p11B", "p12C", "p12D")}
    terminal = {name: {str(x): str(p) for x, p in final_distribution(LOTTERIES[name], gift).items()}
                for name, gift in (("p11A", 1000), ("p11B", 1000), ("p12C", 2000), ("p12D", 2000))}
    return {"evidence": "exact enumeration and deliberately constructed mechanisms; no human outcomes generated",
            "money_means_Israeli_pounds": means, "constructed_weighted_scores": weighted,
            "constructed_reference_scores": reference, "terminal_distributions": terminal}


class NumericalChecks(unittest.TestCase):
    def test_exact_money_means(self):
        self.assertEqual([expected(LOTTERIES[k]) for k in ("p1A", "p1B", "p2C", "p2D")], [2409, 2400, 825, 816])

    def test_common_consequence_for_two_utilities(self):
        # Compute two pairs independently from the lottery definitions.
        for utility in (lambda x: F(x), lambda x: F(x * x, 1000)):
            d1 = expected(LOTTERIES["p1A"], utility) - expected(LOTTERIES["p1B"], utility)
            d2 = expected(LOTTERIES["p2C"], utility) - expected(LOTTERIES["p2D"], utility)
            self.assertEqual(d1, d2)

    def test_common_ratio_scales_expected_utility_difference(self):
        utility = lambda x: F(x * x, 1000)
        d1 = expected(LOTTERIES["p3A"], utility) - expected(LOTTERIES["p3B"], utility)
        d2 = expected(LOTTERIES["p4C"], utility) - expected(LOTTERIES["p4D"], utility)
        self.assertEqual(d2, d1 / 4)

    def test_same_terminal_lotteries(self):
        self.assertEqual(final_distribution(LOTTERIES["p11A"], 1000), final_distribution(LOTTERIES["p12C"], 2000))
        self.assertEqual(final_distribution(LOTTERIES["p11B"], 1000), final_distribution(LOTTERIES["p12D"], 2000))

    def test_constructed_mechanisms_are_separate(self):
        r = calculate()
        w, v = r["constructed_weighted_scores"], r["constructed_reference_scores"]
        self.assertGreater(w["p3B"], w["p3A"])
        self.assertGreater(w["p4C"], w["p4D"])
        self.assertGreater(v["p11B"], v["p11A"])
        self.assertGreater(v["p12C"], v["p12D"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    if args.test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(NumericalChecks)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    print(json.dumps(calculate(), indent=2, ensure_ascii=False))
