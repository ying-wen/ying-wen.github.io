#!/usr/bin/env python3
"""Two-step score, occupancy, baseline variance and critic bias. Standard library only.

Download this single file, then run:
    python3 policy-gradient-walkthrough.py
    python3 policy-gradient-walkthrough.py --json
The policy is frozen. Decimal enumerates all four outcomes; nothing is sampled.
"""
import argparse
import itertools
import json
from decimal import Decimal, localcontext


def calculate(theta="0.7", gamma="0.5"):
    with localcontext() as ctx:
        ctx.prec = 50
        D = Decimal
        theta, gamma = D(theta), D(gamma)
        p = 1 / (1 + (-theta).exp())
        q = 1 - p
        if not (0 < p < 1 and 0 <= gamma <= 1):
            raise ValueError("nondegenerate policy and gamma in [0,1] required")
        values = [gamma * p * q, D(0), q]
        optimum = [gamma * q * q, D(0), p]
        # State IDs restore the Markov environment; the actor still shares one logit.
        successors = {0: 1, 1: 2}
        paths = []
        for a, b in itertools.product((0, 1), repeat=2):
            reward = D(a == 1 and b == 0)
            paths.append({"actions": [a, b], "probability": (p if a else q) * (p if b else q),
                          "reward": reward, "returns": [gamma * reward, reward],
                          "scores": [D(a) - p, D(b) - p], "state1": successors[a]})

        def expect(f, rows=paths):
            return sum((r["probability"] * f(r) for r in rows), D(0))

        def moments(baseline):
            parts = [((r["returns"][0] - baseline[0]) * r["scores"][0],
                      gamma * (r["returns"][1] - baseline[r["state1"]]) * r["scores"][1]) for r in paths]
            mean0 = sum(r["probability"] * g[0] for r, g in zip(paths, parts))
            mean1 = sum(r["probability"] * g[1] for r, g in zip(paths, parts))
            v0 = sum(r["probability"] * (g[0] - mean0) ** 2 for r, g in zip(paths, parts))
            v1 = sum(r["probability"] * (g[1] - mean1) ** 2 for r, g in zip(paths, parts))
            cov = sum(r["probability"] * (g[0] - mean0) * (g[1] - mean1) for r, g in zip(paths, parts))
            return {"baseline": baseline, "mean": mean0 + mean1, "variance0": v0, "variance1": v1,
                    "covariance": cov, "variance": v0 + v1 + 2 * cov,
                    "gradients": [a + b for a, b in parts]}

        def local_variance(state, baseline):
            t = int(state != 0)
            rows = [r for r in paths if state == 0 or r["state1"] == state]
            mass = sum(r["probability"] for r in rows)
            mean = expect(lambda r: (r["returns"][t] - baseline) * r["scores"][t], rows) / mass
            return expect(lambda r: ((r["returns"][t] - baseline) * r["scores"][t] - mean) ** 2, rows) / mass

        states = [{"name": "s0", "mass": D(1), "value": values[0], "actionValues": [D(0), gamma * q]},
                  {"name": "s1(0)", "mass": gamma * q, "value": D(0), "actionValues": [D(0), D(0)]},
                  {"name": "s1(1)", "mass": gamma * p, "value": q, "actionValues": [D(1), D(0)]}]
        for s in states:
            s["localScore"] = sum(prob * score * av for prob, score, av in zip((q, p), (-p, q), s["actionValues"]))
            s["contribution"] = s["mass"] * s["localScore"]

        def td_mean(errors):
            V = [v + e for v, e in zip(values, errors)]
            return expect(lambda r: (gamma * V[r["state1"]] - V[0]) * r["scores"][0]
                          + gamma * (r["reward"] - V[r["state1"]]) * r["scores"][1])

        baselines = [("zero", [D(0)] * 3), ("value", values), ("local-optimum", optimum), ("constant-3", [D(3)] * 3)]
        errors = [[D(0)] * 3, [D(7), D(0), D(0)], [D(0), D(".3"), D(".3")], [D(0), D(0), D(".5")]]
        return {"parameters": {"theta": theta, "gamma": gamma, "p": p, "J": gamma * p * q, "analytic": gamma * p * q * (1 - 2 * p)},
                "trajectories": paths, "states": states, "occupancyGradient": sum(s["contribution"] for s in states), "Z": 1 + gamma,
                "baselineResults": [{"name": name, **moments(b)} for name, b in baselines], "localBaselines": optimum,
                "localVariances": [{"state": i, "zero": local_variance(i, D(0)), "value": local_variance(i, values[i]), "optimal": local_variance(i, optimum[i])} for i in range(3)],
                "criticResults": [{"errors": e, "tdMean": td_mean(e), "mcMean": moments([v + z for v, z in zip(values, e)])["mean"], "bias": gamma * p * q * (e[2] - e[1])} for e in errors],
                "misuse": {"missingOuterDiscount": expect(lambda r: r["returns"][0] * r["scores"][0] + r["returns"][1] * r["scores"][1]),
                           "sameSampleBaseline": expect(lambda r: (r["returns"][0] - r["returns"][0]) * r["scores"][0] + gamma * (r["returns"][1] - r["returns"][1]) * r["scores"][1])}}


def plain(value):
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [plain(v) for v in value]
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--theta", default="0.7")
    parser.add_argument("--gamma", default="0.5")
    args = parser.parse_args()
    data = plain(calculate(args.theta, args.gamma))
    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print("Frozen two-step policy; exhaustive outcomes, no random training")
        print("p, J, dJ/dtheta:", data["parameters"])
        for state in data["states"]:
            print("state / occupancy / gradient contribution:", state["name"], state["mass"], state["contribution"])
        print("Four path probabilities:", [r["probability"] for r in data["trajectories"]])
        for row in data["baselineResults"][:3]:
            print("baseline / mean / episode variance / covariance:", row["name"], row["mean"], row["variance"], row["covariance"])
        print("Local conditional variances:", data["localVariances"])
        print("Critic errors / TD mean / MC mean:")
        for row in data["criticResults"]:
            print(row["errors"], row["tdMean"], row["mcMean"])
        print("Missing outer discount / same-sample baseline:", data["misuse"])
