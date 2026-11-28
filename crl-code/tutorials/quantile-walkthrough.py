#!/usr/bin/env python3
"""Two-step settlement quantiles; independent standard-library calculations.

Save in any empty directory and run python3 quantile-walkthrough.py.
--json reproduces the figure data; --test checks derivatives, CDF integrals,
atom minimizers, terminal rewards and action choices. No random training.
"""
import argparse
from fractions import Fraction as F
import json
import math


def quantile(law, tau):
    """Generalized inverse: first positive-mass outcome with CDF >= tau."""
    if not 0 < tau <= 1:
        raise ValueError("tau outside (0,1]")
    c = 0
    for z, p in sorted(law):
        c += p
        if p and c >= tau - 1e-14:
            return z
    raise ValueError("invalid distribution")


def atom_subgradient(law, theta, tau):
    return [sum(p for z, p in law if z < theta)-tau,
            sum(p for z, p in law if z <= theta)-tau]


def cvar(law, eta=F(1, 2)):
    """Independent variational definition, not JS's inverse-CDF integration."""
    if not 0 < eta <= 1:
        raise ValueError("invalid tail mass")
    return max(b-sum(p*max(0, b-z) for z, p in law)/eta
               for b in {z for z, _ in law})


def summary(atoms, masses=None):
    law = list(zip(atoms, masses or [1/len(atoms)]*len(atoms)))
    return {"mean": sum(z*p for z, p in law), "lower_cvar": cvar(law)}


def cdf_distance(a, b):
    """W1 = area between CDFs in return space, unlike JS quantile-space W1."""
    xs = sorted({z for z, _ in a+b})
    result = 0
    for left, right in zip(xs, xs[1:]):
        fa = sum(p for z, p in a if z <= left)
        fb = sum(p for z, p in b if z <= left)
        result += (right-left)*abs(fa-fb)
    return result


def rho(u, tau, kind="pinball", kappa=1):
    if kind == "pinball":
        return tau*max(u, 0)+(1-tau)*max(-u, 0)
    magnitude = abs(u)
    huber = u*u/2 if magnitude <= kappa else kappa*(magnitude-kappa/2)
    return (tau if u >= 0 else 1-tau)*huber/kappa


def pairwise(theta, taus, target, kind="pinball", kappa=1):
    n, m = len(theta), len(target)
    residuals = [[y-x for y in target] for x in theta]
    loss = sum(rho(y-x, t, kind, kappa) for x, t in zip(theta, taus)
               for y in target)/(n*m)
    law = [(y, 1/m) for y in target]
    if kind == "pinball":
        # CDF probabilities independently derive the expected sample derivative.
        # A sample equal to theta receives the legal subgradient zero.
        head = [sum(p for y, p in law if y < x)
                -t*sum(p for y, p in law if y != x)
                for x, t in zip(theta, taus)]
    else:
        head = []
        for x, t in zip(theta, taus):
            over = sum(p*min(x-y, kappa) for y, p in law if y < x)
            under = sum(p*min(y-x, kappa) for y, p in law if y > x)
            head.append(((1-t)*over-t*under)/kappa)
    return {"kind": kind, "kappa": kappa if kind == "huber" else None,
            "loss": loss, "residuals": residuals,
            "head_gradient": head, "gradient": [g/n for g in head]}


def step(theta, taus, target, rate=4, kind="pinball"):
    d = pairwise(theta, taus, target, kind)
    after = [x-rate*g for x, g in zip(theta, d["gradient"])]
    return dict(d, rate=rate, before=list(theta), target=list(target), after=after,
                after_loss=pairwise(after, taus, target, kind)["loss"])


def targets(reward, successor, gamma=F(1, 2), terminal=False):
    return [reward+gamma*z if not terminal else reward for z in successor]


def select_mean(actions):
    scores = {name: summary(atoms)["mean"] for name, atoms in actions.items()}
    action = "fixed"
    for name, score in scores.items():
        if score > scores[action]+1e-12:
            action = name
    return {"scores": scores, "action": action}


def data():
    taus = [(i+.5)/4 for i in range(4)]
    floating = {"atoms": [.5, 2.5], "masses": [.5, .5]}
    law = list(zip(floating["atoms"], floating["masses"]))
    target, initial = targets(.5, [0, 0, 4, 4]), [1, 1, 2, 2]
    snapshots, current = [], list(initial)
    for k in range(33):
        if k in [0, 1, 4, 32]:
            snapshots.append(dict(updates=k, atoms=list(current), **summary(current),
                                  w1=cdf_distance(list(zip(current, [.25]*4)), law),
                                  loss=pairwise(current, taus, target)["loss"]))
        if k < 32:
            current = step(current, taus, target, 4/math.sqrt(k+1))["after"]
    # Independent root search verifies the closed-form Huber stationary points.
    smooth = []
    for tau in taus:
        left, right = .5, 2.5
        for _ in range(60):
            mid = (left+right)/2
            g = pairwise([mid], [tau], target, "huber")["head_gradient"][0]
            if g > 0:
                right = mid
            else:
                left = mid
        smooth.append((left+right)/2)
    coarse = {"atoms": [.5, 2.5], "masses": [.2, .8]}
    coarse_points = [quantile(list(zip(coarse["atoms"], coarse["masses"])), t) for t in taus]
    bonus = {"fixed": [1.5]*4, "floating": [.6, .6, 2.6, 2.6]}
    return {"kind": "constructed-exact-settlement-quantiles", "random_rollouts": 0,
            "neural_training_steps": 0, "exact_expected_updates": 32,
            "task": {"gamma": .5, "advance": .5, "fixed_final_reward": 2,
                     "floating_final_rewards": [0, 4], "floating_final_masses": [.5, .5],
                     "terminal_future_return": 0}, "taus": taus,
            "true_laws": {"fixed": {"atoms": [1.5], "masses": [1]}, "floating": floating},
            "true_midpoints": {"fixed": [1.5]*4, "floating": target},
            "transition": {"state": "s", "action": "floating", "reward": .5,
                           "next_state": "v", "terminal": False,
                           "frozen_successor": [0, 0, 4, 4], "target": target},
            "sample_step": step(initial, taus, target),
            "single_low_sample": step(initial, taus, [.5]), "snapshots": snapshots,
            "huber": {"kappa": 1, "step": step(initial, taus, target, kind="huber"),
                      "stationary_points": smooth, "stationary_readout": summary(smooth),
                      "at_true": pairwise(target, taus, target, "huber")},
            "atom_conditions": {"at_low": atom_subgradient(law, .5, .125),
                                "median_interval": [.5, 2.5], "canonical_median": quantile(law, .5)},
            "finite_resolution": {"law": coarse, "midpoints": coarse_points,
                                  "true_readout": summary(coarse["atoms"], coarse["masses"]),
                                  "approximate_readout": summary(coarse_points)},
            "bonus": {"advance": .6, "atoms": bonus, "mean_choice": select_mean(bonus),
                      "risk_scores": {a: summary(v)["lower_cvar"] for a, v in bonus.items()},
                      "risk_action": "fixed"},
            "iqn_specified_levels": {"current": [.15, .7], "target": [.2, .4, .9],
                                     "decision": [.1, .3, .6, .8],
                                     "true_target_samples": [.5, .5, 2.5],
                                     "current_true_values": [.5, 2.5]}}


def tests():
    checks = 0

    def close(a, b, tol=1e-9):
        nonlocal checks
        assert abs(a-b) < tol, (a, b)
        checks += 1

    law = [(F(1, 2), F(1, 2)), (F(5, 2), F(1, 2))]
    for tau in [F(1, 8), F(3, 8), F(5, 8), F(7, 8)]:
        point = quantile(law, tau)
        interval = atom_subgradient(law, point, tau)
        assert interval[0] <= 0 <= interval[1]
        checks += 1
    for x in [F(1, 2), F(3, 4), F(3, 2), F(5, 2)]:
        close(sum(p*rho(z-x, F(1, 2)) for z, p in law), F(1, 2))
    # Median endpoint conventions and all interval minimizers are different facts.
    assert quantile(law, F(1, 2)) == F(1, 2)
    checks += 1
    taus = [F(1, 8), F(3, 8), F(5, 8), F(7, 8)]
    target = [F(1, 2)]*2+[F(5, 2)]*2
    d = step([F(1)]*2+[F(2)]*2, taus, target)
    assert d["after"] == [F(5, 8), F(7, 8), F(17, 8), F(19, 8)]
    checks += 1
    for kind in ["pinball", "huber"]:
        for theta in [[.9, 1.2, 1.8, 2.1], [-.2, .8, 2.2, 3.2]]:
            result = pairwise(theta, taus, target, kind)
            for i, g in enumerate(result["gradient"]):
                h, plus, minus = 1e-5, list(theta), list(theta)
                plus[i] += h
                minus[i] -= h
                close((pairwise(plus, taus, target, kind)["loss"]
                       -pairwise(minus, taus, target, kind)["loss"])/(2*h), g)
    for eta in [F(1, 8), F(1, 2), F(3, 4), F(1)]:
        # Independent inverse-CDF integration of the exact two atoms.
        integral = (min(eta, F(1, 2))*F(1, 2)
                    +max(0, eta-F(1, 2))*F(5, 2))/eta
        close(cvar(law, eta), integral)
    for gamma in [F(0), F(1, 2), F(1)]:
        assert targets(F(2), [F(-100), F(100)], gamma, True) == [2, 2]
        checks += 1
    close(summary(data()["snapshots"][-1]["atoms"])["mean"], 1.5)
    for t, point in zip(taus, data()["huber"]["stationary_points"]):
        close(pairwise([point], [t], target, "huber")["head_gradient"][0], 0)
    close(cvar([(F(1, 2), F(1, 5)), (F(5, 2), F(4, 5))]), F(17, 10))
    assert data()["bonus"]["mean_choice"]["action"] == "floating"
    checks += 1
    print(f"{checks} independent quantile checks passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.test:
        tests()
    elif args.json:
        print(json.dumps(data(), ensure_ascii=False, indent=2))
    else:
        d = data()
        print("True midpoint quantiles:", d["true_midpoints"])
        print("One detached-target pinball step:", d["sample_step"]["after"])
        for snapshot in d["snapshots"]:
            print("Exact expected updates:", snapshot)
        print("Huber stationary values (kappa=1):", d["huber"]["stationary_points"])
        print("Finite-N readout:", d["finite_resolution"])
        print("Bonus mean/risk choices:", d["bonus"]["mean_choice"]["action"], d["bonus"]["risk_action"])
        print("No RNG, neural training or performance benchmark.")
