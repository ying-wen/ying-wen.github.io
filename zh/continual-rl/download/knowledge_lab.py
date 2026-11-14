#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ying Wen
"""Small, inspectable checks of predictive knowledge and option-model interfaces.

Original educational code, not a reproduction of Horde, STOMP, or OaK.
Python standard library only; prints JSON and does not write files.

MIT License: Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions: The above copyright notice and this
permission notice shall be included in all copies or substantial portions of the
Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import argparse
import json
import math
import random


def valid_gamma(gamma):
    if not math.isfinite(gamma) or not 0 <= gamma < 1:
        raise ValueError("gamma must be finite and in [0, 1)")
    return gamma


def gvf(seed=0, steps=20000, gamma=0.9, behavior_right=0.6):
    """Tabular one-step IS TD for two target-policy questions.

    States 0 -> 1 -> terminal 2 when moving right; left returns to 0.
    Behavior picks right with probability p; target ALWAYS picks right.
    Each terminal transition naturally starts a new episode at 0; this is NOT
    a single-life task. Both questions see the same transitions; no replay.
    Arrival cumulant = 1 on entering 2; energy cumulant = 1 on every transition.
    Gamma on a terminal transition = 0, to avoid bootstrapping across episodes.
    """
    valid_gamma(gamma)
    if steps < 1 or not 0.1 <= behavior_right <= 1:
        raise ValueError("steps must be positive; behavior_right must be in [0.1, 1]")
    rng = random.Random(seed)
    arrival, energy = [0.0, 0.0], [0.0, 0.0]
    counts = [[0, 0], [0, 0]]  # [left count, right count] per nonterminal state
    state, alpha = 0, 0.03
    for _ in range(steps):
        right = rng.random() < behavior_right
        next_state = state + 1 if right else 0
        terminal = next_state == 2
        counts[state][int(right)] += 1
        rho = 1.0 / behavior_right if right else 0.0
        continuation = 0.0 if terminal else gamma
        for weights, cumulant in ((arrival, float(terminal)), (energy, 1.0)):
            next_value = 0.0 if terminal else weights[next_state]
            delta = cumulant + continuation * next_value - weights[state]
            weights[state] += alpha * rho * delta
        state = 0 if terminal else next_state
    exact = {"arrival": [gamma, 1.0], "energy": [1.0 + gamma, 1.0]}
    learned = {"arrival": arrival, "energy": energy}
    error = max(abs(learned[q][s] - exact[q][s]) for q in exact for s in range(2))
    return {"seed": seed, "steps": steps, "gamma": gamma, "behavior_right": behavior_right,
            "target": "always right", "exact": exact, "learned": learned,
            "max_absolute_error": error, "counts_left_right": counts,
            "protocol": "episodic tabular diagnostic; hand-specified questions; no replay"}


def subtask():
    """Finite undiscounted toy paths, same terminal bonus, different rewards."""
    paths = {"shortcut": {"steps": 2, "reward": -6.0},
             "detour": {"steps": 4, "reward": -1.0}}
    shortest = {name: -p["steps"] for name, p in paths.items()}
    respecting = {name: p["reward"] + 5.0 for name, p in paths.items()}
    return {"gamma": 1.0, "finite_paths_only": True, "terminal_bonus": 5.0,
            "shortest_path_scores": shortest, "reward_respecting_scores": respecting,
            "shortest_choice": max(shortest, key=shortest.get),
            "respecting_choice": max(respecting, key=respecting.get),
            "scope": "illustrates objective design; no learned feature/goal discovery"}


def option_backup(rewards, end_value, gamma):
    valid_gamma(gamma)
    if not rewards:
        raise ValueError("an option must execute at least one primitive step")
    return sum(gamma ** k * r for k, r in enumerate(rewards)) + gamma ** len(rewards) * end_value


def model(gamma=0.9):
    valid_gamma(gamma)
    discounted_mass = (gamma + gamma ** 3) / 2
    nonlinear_expected = ((-1) ** 2 + 1 ** 2) / 2
    nonlinear_at_mean = ((-1 + 1) / 2) ** 2
    # End-state features and duration are correlated; keep their joint moment.
    joint_feature = (gamma * 2.0 + gamma ** 3 * 4.0) / 2
    linear_weight = 3.0
    exact_linear = (gamma * (linear_weight * 2.0) + gamma ** 3 * (linear_weight * 4.0)) / 2
    return {"gamma": gamma, "expected_gamma_to_duration": discounted_mass,
            "correct_terminal_value": 10 * discounted_mass,
            "using_mean_duration": 10 * gamma ** 2,
            "nonlinear_expected_value": nonlinear_expected,
            "nonlinear_value_at_mean": nonlinear_at_mean,
            "linear_exact": exact_linear,
            "linear_via_discounted_feature_model": linear_weight * joint_feature,
            "scope": "exact enumeration; neither fitting a world model nor running a benchmark"}


def planning(gamma=0.9):
    correct = option_backup([-1.0, -1.0, 0.0], 10.0, gamma)
    wrong = -1.0 - gamma + gamma * 10
    rewards, durations = {"A": 3.0, "B": 5.0}, {"A": 2, "B": 5}
    rates = {k: rewards[k] / durations[k] for k in rewards}
    best_rate = max(rates.values())
    # A stale model says a one-step option earns 2; after a change it earns -2.
    # Safe alternative earns 0, all return to the same state: comparison is exact.
    stale, current = {"risky": 2.0, "safe": 0.0}, {"risky": -2.0, "safe": 0.0}
    return {"gamma": gamma, "correct_three_step_backup": correct,
            "incorrect_one_discount_backup": wrong, "average_reward_rates": rates,
            "duration_adjusted_at_best_rate": {k: rewards[k] - best_rate * durations[k] for k in rewards},
            "stale_model_choice": max(stale, key=stale.get),
            "current_model_choice": max(current, key=current.get),
            "scope": "arithmetic diagnostic, not a learned average-reward or model-adaptation algorithm"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=["all", "gvf", "subtask", "model", "planning"], nargs="?", default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--behavior-right", type=float, default=0.6)
    args = parser.parse_args()
    runners = {"gvf": lambda: gvf(args.seed, args.steps, args.gamma, args.behavior_right),
               "subtask": subtask, "model": lambda: model(args.gamma),
               "planning": lambda: planning(args.gamma)}
    try:
        valid_gamma(args.gamma)
        results = {k: run() for k, run in runners.items() if args.experiment in ("all", k)}
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
