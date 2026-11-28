#!/usr/bin/env python3
"""Retention versus adaptation: a deterministic two-parameter diagnostic.

Run: python3 retention_plasticity_walkthrough.py
Only Python's standard library is used. Decimal samplewise gradients are
computed independently of the JavaScript aggregate-gradient implementation.
All evaluation values enumerate the full support; no generalization or RL
performance is estimated. Nothing is written; JSON is printed to stdout.
"""
import json
from decimal import Decimal, localcontext
from fractions import Fraction


def losses(weights, target):
    u, v = weights
    samples = [(v * (u * x) - target * x) ** 2 / 2 for x in (-1, 1)]
    return sum(samples) / 2


def trajectory(initial, target, steps=8):
    weights = [Decimal(str(w)) for w in initial]
    rows = []
    for k in range(steps + 1):
        u, v = weights
        rows.append(dict(k=k, weights=list(map(float, weights)), product=float(u * v),
                         oldLoss=float(losses(weights, 1)), newLoss=float(losses(weights, -1)),
                         targetLoss=float(losses(weights, target))))
        if k < steps:
            gradients = [Decimal(0), Decimal(0)]
            for x in (-1, 1):
                hidden = u * x
                residual = v * hidden - target * x
                gradients[0] += residual * v * x / 2
                gradients[1] += residual * hidden / 2
            weights = [w - g / 4 for w, g in zip(weights, gradients)]
    return rows, weights


def data():
    # Exact rational first steps guard the simultaneous-update convention.
    u, v = Fraction(2), Fraction(1, 2)
    residual = u * v + 1
    next_u, next_v = u - residual * v / 4, v - residual * u / 4
    assert (next_u, next_v) == (Fraction(7, 4), Fraction(-1, 2))
    assert (next_u * next_v + 1) ** 2 / 2 == Fraction(1, 128)
    branches = {}
    with localcontext() as ctx:
        ctx.prec = 60
        for name, initial in dict(balanced=[1, 1], reparameterized=[2, .5], fresh=[1, 0]).items():
            adaptation, checkpoint = trajectory(initial, -1)
            before = list(checkpoint)
            return_probe, _ = trajectory(checkpoint, 1)
            assert checkpoint == before
            branches[name] = dict(initial=initial, adaptation=adaptation, returnProbe=return_probe,
                                  newImprovement=adaptation[0]['newLoss'] - adaptation[-1]['newLoss'],
                                  oldReadOnly=adaptation[-1]['oldLoss'],
                                  returnImprovement=return_probe[0]['oldLoss'] - return_probe[-1]['oldLoss'])
        fresh_old, _ = trajectory([1, 0], 1)
    return dict(branches=branches, freshOldProbe=fresh_old)


if __name__ == '__main__':
    print(json.dumps(data(), indent=2))
