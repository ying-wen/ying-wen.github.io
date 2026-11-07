#!/usr/bin/env python3
"""Four observations, one delayed supervised target; no stochastic training.

Run: python3 recurrent_state_walkthrough.py [--test]
All linear-example values use Fraction. The parameter is fixed within the
sequence; the single displayed update happens only after the final target.
The target, observations, h_0 and (in recurrent modes) readout w are constants.
"""
import argparse
import json
import math
from fractions import Fraction as F


def forward_rtrl(observations, a=F(1, 2), b=F(1), window=None):
    """Forward-mode derivative; detach the prefix without changing its values."""
    h, sa, sb = F(0), F(0), F(0)
    rows = []
    boundary = len(observations) - window if window is not None else -1
    for i, observation in enumerate(observations):
        if i == boundary:
            sa, sb = F(0), F(0)
        old_h = h
        h = a * old_h + b * observation
        sa, sb = old_h + a * sa, observation + a * sb
        rows.append({"t": i + 1, "observation": observation, "old_h": old_h,
                     "h": h, "sensitivity": [sa, sb]})
    return rows


def loss(observations, a=F(1, 2), b=F(1), w=F(1), target=F(1)):
    h = forward_rtrl(observations, a, b)[-1]["h"]
    return (w * h - target) ** 2 / 2


def data():
    observations = [F(1), F(0), F(0), F(0)]
    rows = forward_rtrl(observations)
    final_h, target, w, alpha = rows[-1]["h"], F(1), F(1), F(1, 10)
    residual = w * final_h - target
    modes = {}
    for name, window in [("full", 4), ("last_two", 2), ("detach_each", 1)]:
        sensitivity = forward_rtrl(observations, window=window)[-1]["sensitivity"]
        gradient = [residual * w * s for s in sensitivity]
        modes[name] = {"window": window, "final_h": final_h,
                       "sensitivity": sensitivity, "gradient_theta": gradient,
                       "write_theta": [-alpha * g for g in gradient], "write_w": F(0)}
    modes["readout_only"] = {"final_h": final_h, "gradient_w": residual * final_h,
                              "write_theta": [F(0), F(0)],
                              "write_w": -alpha * residual * final_h}
    # No hidden cue or label is ever passed into the state constructor.
    erased = forward_rtrl([F(0)] * 4)[-1]
    eps = F(1, 10**6)
    finite_difference = [
        (loss(observations, a=F(1, 2) + eps) - loss(observations, a=F(1, 2) - eps)) / (2 * eps),
        (loss(observations, b=F(1) + eps) - loss(observations, b=F(1) - eps)) / (2 * eps),
    ]
    return {"kind": "exact-mechanism", "sampled_runs": 0,
            "config": {"a": F(1, 2), "b": F(1), "w": w, "target": target,
                       "alpha": alpha, "h0": F(0), "finite_difference_epsilon": eps},
            "observations": observations, "rows": rows,
            "prediction": w * final_h, "residual": residual,
            "loss": residual ** 2 / 2, "modes": modes,
            "finite_difference": finite_difference,
            "erased_cue": {"final_h": erased["h"], "sensitivity": erased["sensitivity"],
                           "prediction_for_positive_target": F(0),
                           "prediction_for_negative_target": F(0)}}


def checks():
    d = data()
    assert [r["h"] for r in d["rows"]] == [F(1), F(1, 2), F(1, 4), F(1, 8)]
    assert d["modes"]["full"]["gradient_theta"] == [-F(21, 32), -F(7, 64)]
    assert d["modes"]["last_two"]["gradient_theta"] == [-F(7, 16), F(0)]
    assert d["modes"]["detach_each"]["gradient_theta"] == [-F(7, 32), F(0)]
    assert d["modes"]["readout_only"]["write_theta"] == [0, 0]
    assert d["erased_cue"]["sensitivity"] == [0, 0]
    for actual, expected in zip(d["finite_difference"], d["modes"]["full"]["gradient_theta"]):
        assert abs(actual - expected) < F(1, 10**10)
    # A second, nonlinear recurrence checks the same forward chain rule.
    observations, a, b = [1.0, 0.0, 0.0, 0.0], .8, .3
    h, sa, sb = 0.0, 0.0, 0.0
    for o in observations:
        old = h
        h = math.tanh(a * old + b * o)
        sa, sb = (1-h*h)*(old+a*sa), (1-h*h)*(o+a*sb)
    def nonlinear_loss(a, b):
        h = 0.0
        for o in observations:
            h = math.tanh(a*h+b*o)
        return .5*(h-.8)**2
    eps = 1e-6
    fd = [(nonlinear_loss(a+eps,b)-nonlinear_loss(a-eps,b))/(2*eps),
          (nonlinear_loss(a,b+eps)-nonlinear_loss(a,b-eps))/(2*eps)]
    assert all(abs(g-s*(h-.8)) < 1e-9 for g, s in zip(fd, [sa,sb]))


def floats(value):
    if isinstance(value, F):
        return float(value)
    if isinstance(value, dict):
        return {k: floats(v) for k, v in value.items()}
    if isinstance(value, list):
        return [floats(v) for v in value]
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    if args.test:
        checks()
        print("Exact linear sensitivities and nonlinear finite differences passed.")
    else:
        print(json.dumps(floats(data()), ensure_ascii=False, indent=2))
