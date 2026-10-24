#!/usr/bin/env python3
"""Shared prediction geometry and gradient memory on one two-step corridor.

Run from the tutorial repository:
    python3 tutorials/shared_gradient_walkthrough.py
    python3 tutorials/shared_gradient_walkthrough.py --json

Only standard-library deterministic arithmetic is used. No training is launched.
Task: A --(+2)--> B --(-1)--> terminal, gamma=1; fixed sole action.
Features A=(1,0), B=(.8,.6), terminal=(0,0); alpha=.5, lambda=.5.
Reference: Sutton & Barto (2020 printing), §§9.1–9.3, 12.2, 12.4–12.5;
van Seijen et al. (2016), JMLR 17(145), Algorithm 2.
"""
from __future__ import annotations

import argparse
import json

FEATURES = ((1.0, 0.0), (0.8, 0.6), (0.0, 0.0))
REWARDS = (2.0, -1.0)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def add(a, b, scale=1.0):
    return [x + scale * y for x, y in zip(a, b)]


def predictions(weights):
    return [dot(x, weights) for x in FEATURES[:-1]]


def squared_error(weights):
    """Half the uniformly weighted sum of squared value errors."""
    return sum((v - y) ** 2 for v, y in zip(predictions(weights), (1, -1))) / 4


def mc_steps(alpha=0.5):
    """After both rewards arrive, visit the two complete-return targets in order."""
    w, rows = [0.0, 0.0], []
    for x, target in zip(FEATURES, (1.0, -1.0)):
        before = w[:]
        error = target - dot(x, w)
        w = add(w, x, alpha * error)
        rows.append(dict(before=before, target=target, error=error, after=w[:],
                         predictions=predictions(w), objective=squared_error(w)))
    return rows


def backward(xs=FEATURES, rewards=REWARDS, alpha=0.5, lam=0.5, gamma=1.0,
             initial=(0.0, 0.0), true_online=True):
    """Record an online update; next_value is computed BEFORE changing w."""
    if len(xs) != len(rewards) + 1 or any(len(x) != len(initial) for x in xs):
        raise ValueError("Need one feature per arrival, with consistent dimensions")
    w, e, old_value = list(initial), [0.0] * len(initial), 0.0
    rows = []
    for t, (x, nxt, reward) in enumerate(zip(xs, xs[1:], rewards)):
        value, next_value = dot(w, x), dot(w, nxt)
        delta = reward + gamma * next_value - value
        faded = [gamma * lam * z for z in e]
        coefficient = 1 - alpha * dot(faded, x) if true_online else 1
        e = add(faded, x, coefficient)
        td_increment = [alpha * delta * z for z in e]
        correction = ([alpha * (value - old_value) * (z - xx)
                       for z, xx in zip(e, x)] if true_online else [0.0] * len(e))
        before = w[:]
        w = add(add(w, td_increment), correction)
        rows.append(dict(t=t, before=before, value=value, next_value=next_value,
                         saved_old_value=old_value, delta=delta, faded=faded,
                         coefficient=coefficient, trace=e[:],
                         td_increment=td_increment, correction=correction, after=w[:]))
        old_value = next_value
    return rows


def online_forward(xs=FEATURES, rewards=REWARDS, alpha=0.5, lam=0.5, gamma=1.0,
                   initial=(0.0, 0.0)):
    """Independent slow definition: rebuild each prefix from initial weights.

    n-step bootstrap for arrival j uses the previous prefix's final w[j-1].
    No trace or backward update is used here. Storage grows with the prefix.
    """
    history, rows = [list(initial)], []
    for horizon in range(1, len(rewards) + 1):
        targets, rebuilt = [], list(initial)
        for k in range(horizon):
            total, target = 0.0, 0.0
            for n in range(1, horizon - k + 1):
                total += gamma ** (n - 1) * rewards[k + n - 1]
                bootstrap = gamma ** n * dot(history[k + n - 1], xs[k + n])
                weight = lam ** (n - 1)
                if n < horizon - k:
                    weight *= 1 - lam
                target += weight * (total + bootstrap)
            targets.append(target)
            rebuilt = add(rebuilt, xs[k], alpha * (target - dot(xs[k], rebuilt)))
        rows.append(dict(horizon=horizon, targets=targets, after=rebuilt[:]))
        history.append(rebuilt)
    return rows


def nonlinear_memory():
    """Same rewards, one nonlinear parameter: v(A)=theta^2, v(B)=theta.

    Initial theta=.5, alpha=.5. First TD error=2+.5-.25=2.25,
    so theta becomes 1.625. This is a finite calculation, not a convergence claim.
    """
    theta, alpha, lam = 0.5, 0.5, 0.5
    old_gradient = 2 * theta
    delta = REWARDS[0] + theta - theta * theta
    theta_new = theta + alpha * delta * old_gradient
    return dict(theta_before=theta, theta_after=theta_new, delta=delta,
                old_gradient_A=old_gradient, current_gradient_A=2 * theta_new,
                remembered_trace=lam * old_gradient + 1,
                recomputed_trace=lam * 2 * theta_new + 1)


def calculate():
    mc = mc_steps()
    accumulating, true = backward(true_online=False), backward()
    return dict(
        provenance=dict(kind="exact", task="A --(+2)--> B --(-1)--> terminal",
                        gamma=1, alpha=.5, lam=.5, features=FEATURES,
                        rewards=REWARDS, returns=(1, -1), no_training=True),
        gram=[[dot(x, y) for y in FEATURES[:-1]] for x in FEATURES[:-1]],
        mc=mc, initial_objective=squared_error([0, 0]),
        accumulating=accumulating, true_online=true, forward=online_forward(),
        credit_to_predictions=[dot(x, accumulating[-1]["trace"]) for x in FEATURES[:-1]],
        control=dict(exit_value=0, continue_before=mc[0]["predictions"][0],
                     continue_after=mc[1]["predictions"][0], true_continue_value=1),
        nonlinear=nonlinear_memory(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print inspectable exact-calculation data")
    args = parser.parse_args()
    data = calculate()
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return
    print("A --(+2)--> B --(-1)--> terminal; gamma=1, alpha=lambda=.5")
    print("MC predictions:", [r["predictions"] for r in data["mc"]])
    print("Uniform half-squared error:", data["initial_objective"],
          "->", data["mc"][0]["objective"], "->", data["mc"][1]["objective"])
    print("Accumulating trace:", data["accumulating"][-1]["trace"])
    print("Accumulating final weights:", data["accumulating"][-1]["after"])
    print("True-online weights by prefix:", [r["after"] for r in data["true_online"]])
    print("Independent forward weights:", [r["after"] for r in data["forward"]])
    print("Nonlinear remembered / recomputed trace:",
          data["nonlinear"]["remembered_trace"], data["nonlinear"]["recomputed_trace"])


if __name__ == "__main__":
    main()

