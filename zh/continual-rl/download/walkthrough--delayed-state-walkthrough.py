#!/usr/bin/env python3
"""Exact delayed-cue control reference; Python 3.10+, standard library only.

Each arrival at C draws a NEW independent hidden door mode. The actual JS
environment is checked against the closed-form Fraction program here. This
program uses the declared finite draw list; it does not sample training runs.
The cue gate overwrites activity; it does not reset the environment or b.
"""
import argparse
import itertools
import json
import math
import unittest
from fractions import Fraction as F

EXAMPLE = [(1, 1, 4), (-1, -1, 8), (1, 1, 4)]


def passage(b, cue, z, delay, rho=F(1), alpha=F(1), method="rtrl", window=2):
    """Expand h_J=b*rho**D*c directly, without the JS trace recurrence."""
    attenuation = rho ** delay
    h = b * attenuation * cue
    action = 1 if h >= 0 else -1
    reward = action * z
    sensitivity = F(0) if method == "tbptt" and delay >= window else attenuation * cue
    gradient = (action * h - reward) * action * sensitivity
    return dict(h=h, action=action, reward=reward, sensitivity=sensitivity,
                prediction=action * h, gradient=gradient, usedParameter=b,
                updatedParameter=b - alpha * gradient, delay=delay)


def run(draws=EXAMPLE, b=F(-1, 2), **options):
    decisions, activity = [], []
    step = 0
    rho = options.get("rho", F(1))
    for i, (z, cue, delay) in enumerate(draws):
        if i:
            step += 1  # Actual return from the preceding door to C.
        row = passage(b, cue, z, delay, **options)
        for j in range(delay + 1):
            # This expression independently expands every forward state.
            h = b * rho ** j * cue
            sensitivity = F(0) if options.get("method") == "tbptt" and j >= options.get("window", 2) else rho ** j * cue
            activity.append(dict(step=step+j, place="C" if j == 0 else "J" if j == delay else "M",
                                 h=h, sensitivity=sensitivity, b=b))
        step += delay + 1  # D forward moves, then the chosen door move.
        row.update(step=step-1, rewardStep=step)
        decisions.append(row)
        activity.append(dict(step=step, place="L" if row["action"] == 1 else "R",
                             h=row["h"], sensitivity=row["sensitivity"], b=b))
        b = row["updatedParameter"]
    return dict(b=b, decisions=decisions, activity=activity,
                totalReward=sum(r["reward"] for r in decisions),
                environmentSteps=step, initializationCalls=1, resetCalls=0,
                terminated=False)


def enumerate_paths(delays=(4, 8, 4), accuracy=F(4, 5), **options):
    paths = []
    for pairs in itertools.product(itertools.product((-1, 1), repeat=2), repeat=len(delays)):
        draws = [(z, c, d) for (z, c), d in zip(pairs, delays)]
        probability = math.prod(F(1, 2) * (accuracy if z == c else 1-accuracy) for z, c, _ in draws)
        paths.append(dict(probability=probability, draws=draws, **run(draws, **options)))
    expected = [sum(p["probability"] * p["decisions"][i]["reward"] for p in paths) for i in range(len(delays))]
    return dict(paths=paths, probability=sum(p["probability"] for p in paths),
                expectedRewards=expected, expectedTotal=sum(expected),
                environmentSteps=paths[0]["environmentSteps"])


def data():
    return dict(live=run(), truncated=run(method="tbptt"), decayed=run(rho=F(1, 2)),
                frozen=run(alpha=F(0)), enumeration=enumerate_paths())


def numeric(value):
    if isinstance(value, F):
        return float(value)
    if isinstance(value, dict):
        return {k: numeric(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [numeric(v) for v in value]
    return value


class Checks(unittest.TestCase):
    def test_real_actions_and_cost(self):
        d = run()
        self.assertEqual([x["action"] for x in d["decisions"]], [-1, -1, 1])
        self.assertEqual([x["reward"] for x in d["decisions"]], [-1, 1, 1])
        self.assertEqual([x["rewardStep"] for x in d["decisions"]], [5, 15, 21])
        self.assertEqual(d["totalReward"], 1)
        frozen = run(alpha=F(0))
        self.assertEqual(frozen["decisions"][1]["action"], 1)
        self.assertEqual(frozen["decisions"][1]["reward"], -1)

    def test_window_endpoint(self):
        for K in (1, 2, 4, 8):
            self.assertEqual(passage(F(-1, 2), 1, 1, K, method="tbptt", window=K)["sensitivity"], 0)
            self.assertEqual(passage(F(-1, 2), 1, 1, K-1, method="tbptt", window=K)["sensitivity"], 1)

    def test_fraction_expectation_not_uniform(self):
        d = enumerate_paths()
        self.assertEqual(len(d["paths"]), 64)
        self.assertEqual(d["probability"], 1)
        self.assertEqual(d["expectedRewards"], [F(-3, 5), F(9, 25), F(9, 25)])
        self.assertEqual(d["expectedTotal"], F(3, 25))
        self.assertEqual(enumerate_paths(method="tbptt")["expectedTotal"], F(-9, 5))
        self.assertEqual(enumerate_paths(rho=F(1, 2))["expectedTotal"], F(-9, 5))
        self.assertEqual(enumerate_paths(accuracy=F(1, 2))["expectedTotal"], 0)
        self.assertNotEqual(len(set(p["probability"] for p in d["paths"])), 1)

    def test_sample_loss_finite_difference(self):
        epsilon = 1e-5
        for rho in (F(1), F(1, 2)):
            for D in (1, 2, 4, 8):
                for c, z in itertools.product((-1, 1), repeat=2):
                    for method in ("rtrl", "tbptt"):
                        b = F(-1, 2)
                        row = passage(b, c, z, D, rho=rho, method=method)
                        d, r = row["action"], row["reward"]
                        # A detached boundary is a constant in THIS test function.
                        def loss(value):
                            h = float(row["h"]) if method == "tbptt" and D >= 2 else value * float(rho)**D * c
                            return (d*h-r)**2/2
                        derivative = (loss(float(b)+epsilon)-loss(float(b)-epsilon))/(2*epsilon)
                        self.assertAlmostEqual(derivative, float(row["gradient"]), places=8)

    def test_decay_keeps_information_without_changing_action(self):
        row = passage(F(-1, 2), 1, 1, 4, rho=F(1, 2))
        self.assertEqual(row["h"], F(-1, 32))
        self.assertEqual(row["sensitivity"], F(1, 16))
        self.assertEqual(row["gradient"], F(-33, 512))
        self.assertEqual(row["updatedParameter"], F(-223, 512))
        for D in (1, 4, 16, 100):
            self.assertNotEqual(F(1, 2)**D, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    if args.test:
        unittest.main(argv=["delayed-state-walkthrough.py"], verbosity=2)
    elif args.json:
        print(json.dumps(numeric(data()), indent=2))
    else:
        for name, result in [("hold + full sensitivity", run()), ("hold + K=2", run(method="tbptt")), ("rho=1/2 + full sensitivity", run(rho=F(1, 2)))]:
            print(name)
            for row in result["decisions"]:
                print("  step", row["rewardStep"], "D", row["delay"], "h", row["h"], "S", row["sensitivity"],
                      "door", "L" if row["action"] == 1 else "R", "reward", row["reward"], "b+", row["updatedParameter"])
            print("  real reward sum:", result["totalReward"], "real steps:", result["environmentSteps"], "environment resets: 0")
        result = enumerate_paths()
        print("64 nonuniform weighted paths: E[rewards] =", result["expectedRewards"], "E[sum] =", result["expectedTotal"])
        print("Exact finite-window calculation; no sampled training or lifetime-gradient claim.")
