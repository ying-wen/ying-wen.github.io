#!/usr/bin/env python3
"""A standard-library, exact-arithmetic exploration tutorial.

Run from any directory:
    python3 exploration-information-walkthrough.py
    python3 exploration-information-walkthrough.py --test

The first command prints exact finite-task calculations as JSON (rational values
are also supplied as strings). The second checks an atomic-step Bellman solver
against whole-history enumeration and an independent closed-form derivation.
There is no RNG, training dependency, file output, or free environment reset.

The unknown side theta is fixed in each world. Belief q=P(theta=R), the sensor
laws, and the recovery duration are known. A good route yields 1 immediately and
leads to G, where collection yields 1 every step. A bad route yields 0, reveals
the side, and requires k additional zero-reward steps to get back to J. With k=0
the bad action itself ends at J. H counts every physical step; G is nonterminal.
"""
from fractions import Fraction as F
from functools import lru_cache
from itertools import product
import json
import sys

ACTIONS = ("left", "right", "sign", "screen")


def validate(h, q, k):
    if not isinstance(h, int) or h < 0 or not isinstance(k, int) or k < 0:
        raise ValueError("h and k must be nonnegative integers")
    if not 0 <= q <= 1:
        raise ValueError("q must be a probability")


def belief_update(q, action, symbol=None, reward=None):
    """Only observed data enter this update; no hidden theta argument."""
    if action == "screen":
        return q
    if action == "sign":
        if symbol not in (-1, 1) or (symbol == 1 and q == 0) or (symbol == -1 and q == 1):
            raise ValueError("impossible sign observation")
        return F(symbol == 1)
    if action in ("left", "right"):
        return F((action == "right") == bool(reward))
    return q


def outcomes(q, action, k):
    """Belief-predictive one-step outcomes, integrated over possible worlds."""
    if action == "screen":
        return [(F(1, 2), "J", 0, q, 0) for _ in (-1, 1)]
    if action == "sign":
        return [(weight, "J", 0, F(right), 0) for right, weight in ((False, 1-q), (True, q)) if weight]
    out = []
    for right, weight in ((False, 1-q), (True, q)):
        if not weight:
            continue
        good = (action == "right") == right
        out.append((weight, "G" if good else "D" if k else "J", 0 if good else k, F(right), int(good)))
    return out


@lru_cache(None)
def solve(h, q, k, place="J", remaining=0):
    """Atomic-step recurrence; it does not use the JS closed-form branches."""
    validate(h, q, k)
    if h == 0:
        return F(0), None, {}
    if place == "G":
        return 1 + solve(h-1, q, k, "G", 0)[0], "collect", {}
    if place == "D":
        assert remaining > 0
        return solve(h-1, q, k, "J" if remaining == 1 else "D", remaining-1)[0], "recover", {}
    values = {}
    for action in ACTIONS:
        values[action] = sum(weight*(reward + solve(h-1, after_q, k, after_place, rem)[0])
                             for weight, after_place, rem, after_q, reward in outcomes(q, action, k))
    chosen = max(ACTIONS, key=lambda a: values[a])  # first maximum: same stated tie rule
    return values[chosen], chosen, values


def history(theta, h=6, k=4, q=F(1, 2), first=None, tape=(), disconnect=False):
    """Execute real primitive steps in a specified world and noise tape."""
    place, remaining, total, events, control_q = "J", 0, 0, [], q
    for t in range(h):
        action = solve(h-t, control_q, k, place, remaining)[1]
        if t == 0 and first:
            action = first
        if disconnect and t > 0 and place == "J" and action == "sign":
            action = "right" if control_q > F(1, 2) else "left"
        old_place, old_q, old_control_q, reward = place, q, control_q, 0
        if place == "G":
            assert action == "collect"
            reward = 1
        elif place == "D":
            assert action == "recover"
            remaining -= 1
            place = "J" if remaining == 0 else "D"
        elif action in ("sign", "screen"):
            z = (1 if theta == "R" else -1) if action == "sign" else tape[t]
            q = belief_update(q, action, symbol=z)
            if not (disconnect and action == "sign"):
                control_q = belief_update(control_q, action, symbol=z)
        else:
            good = (action == "right") == (theta == "R")
            reward = int(good)
            q = belief_update(q, action, reward=reward)
            control_q = belief_update(control_q, action, reward=reward)
            place, remaining = ("G", 0) if good else ("D" if k else "J", k)
        total += reward
        events.append({"step": t+1, "from": old_place, "action": action, "reward": reward,
                       "to": place, "recovery_remaining": remaining, "q_before": old_q, "q_after": q,
                       "control_q_before": old_control_q, "control_q_after": control_q})
    return total, events


def enumerate_histories(h=6, k=4, q=F(1, 2), first=None, disconnect=False):
    total, mass, distribution = F(0), F(0), {}
    for theta, p in (("L", 1-q), ("R", q)):
        if not p:
            continue
        for tape in product((-1, 1), repeat=h):
            weight = p / 2**h
            reward, _ = history(theta, h, k, q, first, tape, disconnect)
            total += weight*reward
            mass += weight
            distribution[reward] = distribution.get(reward, F(0)) + weight
    return {"value": total, "mass": mass, "outcomes": distribution}


def closed_form(h, q, k):
    if not h:
        return F(0)
    wrong = max(0, h-1-k)
    return max((1-q)*h+q*wrong, q*h+(1-q)*wrong, h-1)


def risk(p_plus, mean):
    return (1-p_plus)*(-1-mean)**2 + p_plus*(1-mean)**2


def tests():
    for h in range(9):
        for k in range(7):
            for q in (F(0), F(1, 4), F(1, 2), F(3, 4), F(1)):
                assert solve(h, q, k)[0] == closed_form(h, q, k)
    for h in range(7):
        for k in (0, 1, 4, 8):
            for first in (None, *ACTIONS):
                actual = enumerate_histories(h=h, k=k, first=first)
                expected = F(0) if not h else solve(h, F(1, 2), k)[0] if first is None else solve(h, F(1, 2), k)[2][first]
                assert actual["mass"] == 1 and actual["value"] == expected
    assert solve(6, F(1, 2), 4)[2] == {"left": F(7, 2), "right": F(7, 2), "sign": 5, "screen": 4}
    assert solve(6, F(1, 2), 0)[0] == F(11, 2)
    assert solve(1, F(1, 2), 4)[0] == F(1, 2)
    assert solve(6, F(1), 4)[0] == 6
    assert enumerate_histories(first="sign", disconnect=True)["value"] == F(5, 2)
    _, stale = history("R", first="sign", tape=(-1,)*6, disconnect=True)
    assert stale[0]["q_after"] == 1 and stale[0]["control_q_after"] == F(1, 2)
    assert stale[1]["action"] == "left"
    _, failure = history("R", first="left", tape=(-1,)*6)
    assert [e["action"] for e in failure] == ["left", "recover", "recover", "recover", "recover", "right"]
    assert failure[0]["q_after"] == 1 and failure[4]["to"] == "J"
    assert risk(F(1, 2), 0) == 1 and risk(F(1, 2), 1) == 2
    assert (1-1)**2 == 0  # fitting the current sample is not fresh-frame progress
    assert sum(F(1, 4)*(new-old)**2 for old, new in product((-1, 1), repeat=2)) == 2
    try:
        belief_update(F(0), "sign", symbol=1)
    except ValueError:
        pass
    else:
        raise AssertionError("zero-probability evidence was accepted")
    print("PASS: atomic Fraction recursion, 315 closed-form cases, 140 whole-history checks, recovery clock, information consumer, and expected-failure controls")


def report():
    matrix = []
    for h in range(9):
        for k in range(7):
            for q in (F(0), F(1, 4), F(1, 2), F(3, 4), F(1)):
                value, action, values = solve(h, q, k)
                matrix.append({"horizon": h, "recovery": k, "prior": q, "value": value, "action": action, "values": values})
    choices = {a: enumerate_histories(first=a) for a in ACTIONS}
    trajectories = {name: {"total": history(theta, first=first, tape=(-1,)*6)[0],
                           "events": history(theta, first=first, tape=(-1,)*6)[1]}
                    for name, theta, first in (("sign_L", "L", "sign"), ("sign_R", "R", "sign"), ("direct_L", "L", "left"), ("direct_R", "R", "left"), ("screen_R", "R", "screen"))}
    return {"matrix": matrix, "choices": choices, "trajectories": trajectories,
            "exact_totals": {a: str(v["value"]) for a, v in choices.items()},
            "disconnected_consumer": enumerate_histories(first="sign", disconnect=True)}


if __name__ == "__main__":
    if "--test" in sys.argv:
        tests()
    else:
        print(json.dumps(report(), default=lambda x: float(x) if isinstance(x, F) else str(x), indent=2))
