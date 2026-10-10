#!/usr/bin/env python3
"""Recurrent replay: independent nested unroll and expanded chain rule.

Python 3.10+, standard library only. Fixed records and given weights; no training.
Run from any directory: python3 recurrent-replay-walkthrough.py --test / --json
"""
import argparse
import json
import math

SETTINGS = {"old": {"w": -.8, "rho": .8}, "current": {"w": .8, "rho": .8},
            "target": {"w": .6, "rho": .7}, "gamma": .9, "alpha": .01,
            "decisionIndex": 4, "burnStart": 2, "learnStart": 4}


def nested(xs, p, initial=0.0):
    """Re-evaluate the entire prefix, instead of carrying a recurrent derivative."""
    if not xs:
        return initial
    return math.tanh(p["rho"] * nested(xs[:-1], p, initial) + p["w"] * xs[-1])


def expanded_rows(xs, p, initial=0.0, detach=(), resets=()):
    rows, forward_start, derivative_start = [], 0, 0
    for i, x in enumerate(xs):
        if i in resets:
            forward_start = derivative_start = i
            initial = 0.0
        if i in detach:
            derivative_start = i
        segment = xs[forward_start:i+1]
        previous = nested(segment[:-1], p, initial)
        h = nested(segment, p, initial)
        # Each summand is one parameter's direct contribution followed by the
        # product of local Jacobians from that node to the current output.
        d_w, d_rho = 0.0, 0.0
        for j in range(max(forward_start, derivative_start), i + 1):
            product = math.prod(1 - nested(xs[forward_start:k+1], p, initial)**2
                                for k in range(j, i + 1)) * p["rho"]**(i-j)
            d_w += xs[j] * product
            d_rho += nested(xs[forward_start:j], p, initial) * product
        rows.append({"index": i, "x": x, "previous": previous, "h": h,
                     "dw": d_w, "drho": d_rho, "jacobian": 1-h*h,
                     "resetBefore": i in resets, "detachedBefore": i in detach})
    return rows


def q(h):
    return {"L": 1 + 3*h, "R": 1 - 3*h}


def action(h):
    return "L" if q(h)["L"] >= q(h)["R"] else "R"


def reward(parcel, chosen):
    return 4 if parcel == chosen else -2


def target(r, terminal, online=None, frozen=None):
    if terminal:
        return {"y": r, "nextAction": None, "continuation": 0}
    chosen = action(online)
    tail = q(frozen)[chosen]
    return {"y": r + SETTINGS["gamma"]*tail, "nextAction": chosen, "continuation": tail}


def terminal_loss(row, chosen="R", r=-2):
    sign = 1 if chosen == "L" else -1
    prediction = q(row["h"])[chosen]
    error = prediction - r
    gradient = {"w": error*3*sign*row["dw"], "rho": error*3*sign*row["drho"]}
    return {"action": chosen, "reward": r, "y": r, "q": prediction,
            "residual": error, "loss": .5*error*error, "gradient": gradient,
            "updated": {k: SETTINGS["current"][k] - SETTINGS["alpha"]*gradient[k]
                        for k in ("w", "rho")}}


def replay(cue):
    xs = [cue, 0, 0, 0, 0]
    old = expanded_rows(xs, SETTINGS["old"])
    current = expanded_rows(xs, SETTINGS["current"])
    frozen = expanded_rows(xs, SETTINGS["target"])
    variants = {
        "full": current,
        "prefixDetached": expanded_rows(xs, SETTINGS["current"], detach=(4,)),
        "storedBurnDetached": expanded_rows(xs[2:], SETTINGS["current"], old[1]["h"], detach=(2,)),
        "storedBurnAttached": expanded_rows(xs[2:], SETTINGS["current"], old[1]["h"]),
        "zeroBurn": expanded_rows(xs[2:], SETTINGS["current"], detach=(2,)),
        "zeroAtLearn": expanded_rows(xs[4:], SETTINGS["current"]),
        "storedAtLearn": expanded_rows(xs[4:], SETTINGS["current"], old[3]["h"]),
    }
    parcel = "L" if cue == 1 else "R"
    recorded = action(old[4]["h"])
    recorded_reward = reward(parcel, recorded)
    results = {}
    for name, rows in variants.items():
        last = rows[-1]
        chosen = action(last["h"])
        results[name] = {"rows": rows, "h": last["h"], "dw": last["dw"],
                         "drho": last["drho"], "q": q(last["h"]), "action": chosen,
                         "freshReward": reward(parcel, chosen),
                         "stateError": abs(last["h"] - current[4]["h"]),
                         "sample": terminal_loss(last, recorded, recorded_reward)}
    return {"cue": cue, "observations": xs, "old": old, "current": current, "target": frozen,
            "recordedAction": recorded, "recordedReward": recorded_reward, "oldAction": recorded,
            "currentAction": action(current[4]["h"]), "oldFreshReward": recorded_reward,
            "currentFreshReward": reward(parcel, action(current[4]["h"])), "results": results,
            "bootstrap": {"prediction": 1+3*current[3]["h"]**2,
                          **target(0, False, current[4]["h"], frozen[4]["h"]),
                          "wrongSharedStateTarget": target(0, False, current[4]["h"], current[4]["h"])["y"]}}


def initial_errors(rho, steps=8):
    p = {"w": .8, "rho": rho}
    start = math.tanh(.8)
    rows = []
    for b in range(steps+1):
        ref, stored, zero = (nested([0]*b, p, h) for h in (start, -start, 0.0))
        rows.append({"B": b, "reference": ref, "stored": stored, "zero": zero,
                     "storedError": abs(ref-stored), "zeroError": abs(ref-zero),
                     "storedAction": action(stored), "referenceAction": action(ref)})
    return rows


def data():
    branches = [replay(-1), replay(1)]
    positive = branches[1]
    return {"settings": SETTINGS, "positive": positive, "branches": branches,
            "initialErrors": {"contractive": initial_errors(.8), "noncontractive": initial_errors(1.2)},
            "missingCue": {"observedSuffix": [0, 0, 0],
                           "zeroStates": [b["results"]["zeroBurn"]["h"] for b in branches],
                           "zeroActions": [b["results"]["zeroBurn"]["action"] for b in branches],
                           "meanFreshReward": sum(.5*b["results"]["zeroBurn"]["freshReward"] for b in branches),
                           "fullMeanFreshReward": sum(.5*b["results"]["full"]["freshReward"] for b in branches)},
            "boundaries": {
                "chunk": expanded_rows([1, 0, 0, 0, 0], SETTINGS["current"], detach=(2,)),
                "falseReset": expanded_rows([1, 0, 0, 0, 0], SETTINGS["current"], resets=(2,)),
                "newEpisode": expanded_rows([1, 0, -1, 0], SETTINGS["current"], resets=(2,)),
                "chunkTarget": positive["bootstrap"]["y"], "falseTerminalTarget": 0,
                "realTerminalTarget": -2},
            "scope": "Fixed nonlinear recurrent Q mechanism; old/current/target weights are given; no random training or benchmark reproduction."}


def self_test():
    near = lambda a, b: math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-7)
    xs, p, epsilon = [1, 0, 0, 0, 0], SETTINGS["current"], 1e-6
    full = replay(1)["results"]["full"]
    detached = replay(1)["results"]["prefixDetached"]
    assert full["h"] == detached["h"] and detached["dw"] == 0
    for key, field in (("w", "dw"), ("rho", "drho")):
        plus, minus = dict(p), dict(p)
        plus[key] += epsilon
        minus[key] -= epsilon
        numerical = (nested(xs, plus)-nested(xs, minus))/(2*epsilon)
        assert near(numerical, full[field])
        loss = lambda param: .5*(q(nested(xs, param))["R"]+2)**2
        numerical_loss = (loss(plus)-loss(minus))/(2*epsilon)
        assert near(numerical_loss, full["sample"]["gradient"][key])
        # The actual boundary number is held fixed when differentiating a
        # detached function. Recomputing it would silently restore full BPTT.
        boundary = nested(xs[:4], p)
        local = (nested([0], plus, boundary)-nested([0], minus, boundary))/(2*epsilon)
        assert near(local, detached[field])
    d = data()
    assert d["positive"]["oldAction"] == "R" and d["positive"]["currentAction"] == "L"
    assert d["positive"]["recordedReward"] == -2 and d["positive"]["currentFreshReward"] == 4
    assert d["missingCue"]["zeroStates"] == [0, 0]
    assert d["missingCue"]["meanFreshReward"] == 1 and d["missingCue"]["fullMeanFreshReward"] == 4
    assert d["boundaries"]["chunk"][-1]["h"] != 0
    assert d["boundaries"]["falseReset"][-1]["h"] == 0
    assert d["boundaries"]["newEpisode"][2]["h"] == -math.tanh(.8)
    assert d["boundaries"]["falseTerminalTarget"] == 0 < d["boundaries"]["chunkTarget"]
    for row in d["initialErrors"]["contractive"]:
        assert row["storedError"] <= .8**row["B"] * 2*math.tanh(.8) + 1e-12
        assert row["storedAction"] == "R" and row["referenceAction"] == "L"
    assert d["initialErrors"]["noncontractive"][-1]["storedError"] > 1.3
    for cue in (-1, 1):
        # Including the cue in zero-initialized burn-in exactly recovers the
        # given current-parameter prefix, even if its gradient is detached.
        full_prefix = expanded_rows([cue, 0, 0, 0, 0], p, detach=(4,))
        assert full_prefix[-1]["h"] == replay(cue)["current"][4]["h"]
    print("PASS: expanded nonlinear gradients, fixed-boundary differences, two histories, targets, resets, and burn-in failures.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.test:
        self_test()
    elif args.json:
        print(json.dumps(data(), ensure_ascii=False, indent=2))
    else:
        d = data()
        print("Given warehouse record: cue +1, four grey frames, old actor collects R and observes -2.")
        print("Current actor with the full cue history would collect L and receive +4 in this deterministic branch.")
        for name, row in d["positive"]["results"].items():
            print(f"{name:22s} h={row['h']:+.6f} action={row['action']} "
                  f"d(h)/d(w,rho)=({row['dw']:+.6f},{row['drho']:+.6f}) "
                  f"sample gradient={row['sample']['gradient']}")
        print("Nonterminal target using the target network's own state:", d["positive"]["bootstrap"])
        print("Missing-cue histories:", d["missingCue"])
        print("Only deterministic replay diagnostics; no environment collection or training run.")


if __name__ == "__main__":
    main()
