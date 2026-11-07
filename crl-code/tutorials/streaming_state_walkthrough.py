#!/usr/bin/env python3
"""Exact, finite learner-state walkthrough; standard library only.

The fixed recurrence h_next = h/2 + observation is a feature constructor,
not a trainable RNN. A scalar linear predictor uses h as a detached feature.
All arithmetic is Fraction arithmetic. The three supplied transitions are a
mechanism example, not training runs or evidence of control performance.
Run with --test to check restoration, reset timing, and replay permissions.
"""
from dataclasses import dataclass, replace
from fractions import Fraction as F
import argparse
import json


CONFIG = {"gamma": F(9, 10), "lam": F(4, 5), "alpha": F(1, 10),
          "beta": F(9, 10), "eps": F(1, 10), "retention": F(1, 2)}
STREAM = [{"next_observation": F(0), "reward": F(2), "terminal": False},
          {"next_observation": F(0), "reward": F(0), "terminal": False},
          {"next_observation": F(0), "reward": F(1), "terminal": True}]


@dataclass(frozen=True)
class Learner:
    w: F = F(0)
    h: F = F(1)  # Initial observation is 1, following h_previous = 0.
    z: F = F(0)
    max_v: F = F(0)
    gamma_in: F = F(0)
    k: int = 0
    cursor: int = 0


def checkpoint(state):
    """Save all persistent state plus the next stream position, not past data."""
    return {name: str(value) if isinstance(value, F) else value
            for name, value in vars(state).items()}


def restore(saved):
    fields = {name: F(value) if name not in ("k", "cursor") else value
              for name, value in saved.items()}
    return Learner(**fields)


def step(old, transition, reset_before_terminal=False):
    """Compute with old weights; reset episode-local state AFTER its update."""
    c = CONFIG
    terminal = transition["terminal"]
    next_h = F(0) if terminal else c["retention"] * old.h + transition["next_observation"]
    gamma_out = F(0) if terminal else c["gamma"]
    delta = transition["reward"] + gamma_out * old.w * next_h - old.w * old.h
    # The deliberately wrong branch tests erasure BEFORE the terminal reward.
    previous_z = F(0) if terminal and reset_before_terminal else old.z
    z = old.gamma_in * c["lam"] * previous_z + old.h
    increment = delta * z
    max_v = max(c["beta"] * old.max_v, abs(increment))
    displacement = c["alpha"] * increment / (max_v + c["eps"])
    after = Learner(old.w + displacement, F(0) if terminal else next_h,
                    F(0) if terminal else z, max_v, gamma_out,
                    old.k + 1, old.cursor + 1)
    row = {"t": old.cursor, "k": after.k, "h": old.h, "next_h": next_h,
           "old_w": old.w, "reward": transition["reward"], "delta": delta,
           "trace_used": z, "increment": increment, "max_v": max_v,
           "displacement": displacement, "w": after.w,
           "terminal": terminal, "after": vars(after)}
    return after, row


def suffix(start, reset_before_terminal=False):
    state, rows = start, []
    for transition in STREAM[start.cursor:]:
        state, row = step(state, transition, reset_before_terminal)
        rows.append(row)
    return {"rows": rows, "final": vars(state)}


def permissions():
    """Same frozen feature transitions, fixed-step TD(0); only data reuse varies.

    The replay branch deliberately retains feature-transition records. The
    strict branch uses each record on arrival and discards it; rows are output
    diagnostics kept by this demonstration, not input to its learner.
    """
    features = [(F(1), F(1, 2), F(2), F(9, 10)),
                (F(1, 2), F(1, 4), F(0), F(9, 10)),
                (F(1, 4), F(0), F(1), F(0))]
    w = F(0)
    for x, next_x, reward, gamma in features:
        w += F(1, 10) * (reward + gamma * w * next_x - w * x) * x
    x, next_x, reward, gamma = features[0]
    replay_delta = reward + gamma * w * next_x - w * x
    replay_w = w + F(1, 10) * replay_delta * x
    return {"strict_w": w, "replay_delta": replay_delta, "replay_w": replay_w,
            "environment_transitions": 3, "strict_updates": 3,
            "replay_updates": 4, "strict_retained_transitions": 0,
            "replay_retained_transitions": 3}


def compute():
    first, first_row = step(Learner(), STREAM[0])
    saved = checkpoint(first)
    resumed = restore(json.loads(json.dumps(saved)))
    starts = {"full": resumed, "clear_trace": replace(resumed, z=F(0)),
              "clear_scale": replace(resumed, max_v=F(0)),
              "clear_activity": replace(resumed, h=F(0)),
              "clear_weight": replace(resumed, w=F(0)),
              "weights_only": Learner(w=resumed.w, h=F(0),
                                      gamma_in=resumed.gamma_in, k=1, cursor=1)}
    return {"kind": "exact-mechanism", "config": CONFIG,
            "stream": STREAM, "first": first_row, "checkpoint": vars(first),
            "branches": {name: suffix(start) for name, start in starts.items()},
            "wrong_terminal_reset": suffix(resumed, True),
            "permissions": permissions(), "random_rollouts": 0}


def to_numbers(value):
    if isinstance(value, F):
        return float(value)
    if isinstance(value, dict):
        return {key: to_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_numbers(item) for item in value]
    return value


def test():
    d = compute()
    saved = d["checkpoint"]
    assert saved["w"] == F(2, 21) and saved["h"] == F(1, 2)
    assert saved["z"] == 1 and saved["max_v"] == 2
    full = d["branches"]["full"]
    assert suffix(Learner())["rows"] == [d["first"], *full["rows"]]
    second = full["rows"][0]
    assert second["delta"] == -F(11, 420)
    assert second["trace_used"] == F(61, 50)
    assert second["increment"] == -F(671, 21000)
    assert second["displacement"] == -F(671, 399000)
    assert d["branches"]["clear_trace"]["rows"][0]["displacement"] == -F(11, 15960)
    assert d["branches"]["clear_scale"]["rows"][0]["displacement"] == -F(671, 27710)
    assert d["branches"]["clear_activity"]["rows"][0]["displacement"] == 0
    assert d["branches"]["clear_weight"]["rows"][0]["displacement"] == 0
    assert d["branches"]["weights_only"]["final"]["w"] == F(2, 21)
    assert full["final"]["h"] == 0 and full["final"]["z"] == 0
    assert full["final"]["max_v"] > 0 and full["final"]["k"] == 3
    assert d["wrong_terminal_reset"]["rows"][-1]["trace_used"] == F(1, 4)
    assert full["rows"][-1]["trace_used"] == F(5642, 5000)
    assert d["permissions"]["strict_w"] == F(141451, 640000)
    assert d["permissions"]["replay_w"] > d["permissions"]["strict_w"]
    print("PASS: exact restore, individual reset, boundary timing, and data reuse")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    if args.test:
        test()
    else:
        print(json.dumps(to_numbers(compute()), ensure_ascii=False, indent=2))
