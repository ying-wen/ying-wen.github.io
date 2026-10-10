#!/usr/bin/env python3
"""One-file exact warehouse POMDP. Python standard library; no sampling.

Run: python3 belief-planning-walkthrough.py
     python3 belief-planning-walkthrough.py --json
     python3 belief-planning-walkthrough.py --test
The supplied calibration counts are a constructed teaching dataset, not a run.
"""
import argparse
import itertools
import json
from fractions import Fraction as F

LOCATIONS = ["L", "C", "R"]
B = [F(1, 2), F(0), F(1, 2)]
COST = F(2, 5)

def sensor(q=F(4, 5)):
    if not 0 <= q <= 1:
        raise ValueError("sensor accuracy must be in [0,1]")
    return [[q if s == o else (1-q)/2 for o in range(3)] for s in range(3)]

def valid(b):
    if len(b) != 3 or min(b) < 0 or sum(b) != 1:
        raise ValueError("normalized belief required")

def posterior(b, o, q=F(4, 5)):
    valid(b)
    joint = [b[s]*sensor(q)[s][o] for s in range(3)]
    evidence = sum(joint)
    if not evidence:
        raise ValueError("zero evidence")
    return evidence, [v/evidence for v in joint]

def terminal(b):
    valid(b)
    values = [6*p-2 for p in b]
    value = max(values)
    return {"action": values.index(value), "value": value, "values": values}

def scan_backup(b, q=F(4, 5), cost=COST):
    valid(b)
    if cost < 0:
        raise ValueError("negative scan cost")
    branches, gross = [], F(0)
    for o in range(3):
        probability = sum(b[s]*sensor(q)[s][o] for s in range(3))
        if not probability:
            branches.append(dict(observation=o, evidence=F(0), belief=None, action=None, value=F(0), values=None))
            continue
        evidence, belief = posterior(b, o, q)
        choice = terminal(belief)
        branches.append(dict(observation=o, evidence=evidence, belief=belief, **choice))
        gross += evidence*choice["value"]
    return dict(q=q, cost=cost, branches=branches, gross=gross, value=gross-cost)

def plan_value(b, plan, q=F(4, 5), cost=COST):
    """Independent enumeration of all hidden-state/signal outcomes."""
    valid(b)
    if len(plan) != 3 or any(a not in range(3) for a in plan):
        raise ValueError("three conditional collection actions required")
    return -cost + sum(b[s]*sensor(q)[s][o]*(4 if plan[o] == s else -2)
                       for s, o in itertools.product(range(3), repeat=2))

def enumerate_plans(b, q=F(4, 5), cost=COST):
    plans = [dict(plan=list(p), value=plan_value(b, p, q, cost))
             for p in itertools.product(range(3), repeat=3)]
    value = max(p["value"] for p in plans)
    return dict(count=len(plans), value=value, best=[p for p in plans if p["value"] == value])

def walkthrough_data():
    centered = [F(0), F(1), F(0)]
    counts = [[8, 1, 1], [1, 8, 1], [1, 1, 8]]
    total = sum(sum(row) for row in counts)
    correct = sum(counts[i][i] for i in range(3))
    fitted = F(correct, total)
    scan = scan_backup(B, fitted)
    return dict(task=dict(horizon=2, gamma=1, scanCost=COST, successReward=4,
                          failureReward=-2, scanAccuracy=F(4, 5)),
                locations=LOCATIONS, positions=[-1, 0, 1], observationMatrix=sensor(),
                ambiguous=B, centered=centered, means=[F(0), F(0)],
                direct=terminal(B), centeredDirect=terminal(centered), scan=scan,
                oldScan=scan_backup(B, F(2, 5)),
                openLoop=[dict(action=a, value=plan_value(B, [a]*3)) for a in range(3)],
                enumeration=enumerate_plans(B), calibrationCounts=counts,
                fit=dict(total=total, correct=correct, q=fitted),
                control=dict(oldChosen="collect-L", oldPredicted=1, oldActual=1,
                             calibratedChosen="scan", calibratedActual=scan["value"], loss=scan["value"]-1),
                accuracyCurve=[dict(q=F(1, 3)+F(2*i, 60), value=scan_backup(B, F(1, 3)+F(2*i, 60))["value"])
                               for i in range(21)])

def tests():
    assert scan_backup(B)["value"] == F(27, 10)
    assert scan_backup(B)["gross"] == F(31, 10)
    assert plan_value(B, [0, 0, 0]) == F(3, 5)
    assert terminal(B)["value"] == 1
    assert scan_backup(B, F(2, 5))["value"] == F(9, 10)
    assert posterior(B, 0)[1] == [F(8, 9), F(0), F(1, 9)]
    assert enumerate_plans(B)["value"] == scan_backup(B)["value"]
    # Verify all rational beliefs on a small simplex, including boundary beliefs.
    for n in range(1, 6):
        for left in range(n+1):
            for center in range(n-left+1):
                b = [F(left, n), F(center, n), F(n-left-center, n)]
                for q in [F(0), F(1, 3), F(2, 5), F(4, 5), F(1)]:
                    assert enumerate_plans(b, q)["value"] == scan_backup(b, q)["value"]
    try:
        posterior([F(0), F(1), F(0)], 0, F(1))
        raise AssertionError("zero evidence must be rejected")
    except ValueError:
        pass
    print("PASS: exact fractions, 27-plan enumeration, simplex and zero-evidence checks")

def as_numbers(value):
    if isinstance(value, F):
        return float(value)
    if isinstance(value, list):
        return [as_numbers(x) for x in value]
    if isinstance(value, dict):
        return {k: as_numbers(v) for k, v in value.items()}
    return value

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    if args.test:
        tests()
    elif args.json:
        print(json.dumps(as_numbers(walkthrough_data()), indent=2, ensure_ascii=False))
    else:
        d = walkthrough_data()
        print("Observation: junction, hidden package position is unavailable to the agent.")
        print("Direct expected return:", terminal(B)["value"])
        print("Fixed scan->L expected return:", plan_value(B, [0, 0, 0]))
        for branch in d["scan"]["branches"]:
            print("Signal", LOCATIONS[branch["observation"]], "probability", branch["evidence"],
                  "posterior", branch["belief"], "collect", LOCATIONS[branch["action"]])
        print("Conditional scan expected return:", d["scan"]["value"])
        print("27-plan check:", d["enumeration"])
        print("Given old q=2/5 chooses direct; calibrated q=4/5 chooses scan; loss = 17/10.")
