#!/usr/bin/env python3
"""Four observations, one actual online parameter update, three derivatives.

Run from any directory: python3 state-online-walkthrough.py [--test]
Python 3.10+ standard library only. Default output is JSON of exact Fraction
calculations converted to floats. --test adds independent finite differences.
No random training, environment reset, or externally supplied parameter schedule
is used to create the main trajectory. Replay is an explicitly labeled analysis.
"""
import argparse
from fractions import Fraction as F
import json
import math

OBSERVATIONS = (1, 0, 0, 0)


def online(a=F(1, 2), b=F(1), alpha=F(1, 2), early_target=F(1)):
    """Execute forward state, carried trace, then the t=2 parameter write."""
    h, sa, sb = F(0), F(0), F(0)
    rows = []
    for t, observation in enumerate(OBSERVATIONS, 1):
        old_h, old_sa, old_sb = h, sa, sb
        used = (a, b)
        h = a * old_h + b * observation
        sa, sb = old_h + a * old_sa, observation + a * old_sb
        gradient, write = None, (F(0), F(0))
        if t == 2:
            gradient = ((h - early_target) * sa, (h - early_target) * sb)
            write = tuple(-alpha * g for g in gradient)
            a, b = a + write[0], b + write[1]
        rows.append(dict(t=t, used=used, h=h, trace=(sa, sb), gradient=gradient,
                         write=write, after=(a, b)))
    return dict(rows=rows, current=(a, b), h=h, trace=(sa, sb))


def scheduled_forward(schedule, offset=(0, 0)):
    """Freeze recorded parameter values, then shift every occurrence equally."""
    h = 0
    for observation, (a, b) in zip(OBSERVATIONS, schedule):
        h = (a + offset[0]) * h + (b + offset[1]) * observation
    return h


def learner_polynomial(a, b, alpha, target=1):
    """Independent oracle: eliminate both the trace and the update loop."""
    return (a - alpha * (a * b - target) * b) ** 2 * a * b


def derivative(function, point, coordinate, epsilon=1e-6):
    plus, minus = list(point), list(point)
    plus[coordinate] += epsilon
    minus[coordinate] -= epsilon
    return (function(*plus) - function(*minus)) / (2 * epsilon)


def results():
    actual = online()
    a, b = actual['current']
    replay_h = a ** 3 * b
    # Algebraic derivatives of the explicit learner_polynomial, independently
    # checked below by finite differences of the sequential learner itself.
    initial_derivative = (F(15, 16), F(9, 32))
    modes = {
        'schedule': dict(h=actual['h'], derivative=actual['trace']),
        'replay': dict(h=replay_h, derivative=(3 * a * a * b, a ** 3)),
        'learner': dict(h=actual['h'], derivative=initial_derivative),
    }
    for mode in modes.values():
        mode['gradient'] = tuple((mode['h'] - 1) * s for s in mode['derivative'])
    return dict(kind='exact-mechanism', sampled_runs=0, actual=actual, modes=modes,
                alpha_derivative=F(3, 8), loss_alpha_derivative=F(-69, 256))


def tests():
    d = results()
    actual = d['actual']
    assert [r['h'] for r in actual['rows']] == [F(1), F(1, 2), F(3, 8), F(9, 32)]
    assert actual['current'] == (F(3, 4), F(9, 8))
    assert actual['rows'][1]['gradient'] == (F(-1, 2), F(-1, 4))
    schedule = [tuple(map(float, r['used'])) for r in actual['rows']]
    current = tuple(map(float, actual['current']))
    functions = {
        'schedule': lambda da, db: scheduled_forward(schedule, (da, db)),
        'replay': lambda da, db: scheduled_forward([current] * 4, (da, db)),
        'learner': lambda da, db: online(.5 + da, 1 + db, .5)['h'],
    }
    checks = 0
    for name, function in functions.items():
        for i in range(2):
            for epsilon in (1e-4, 1e-5, 1e-6):
                assert math.isclose(derivative(function, (0, 0), i, epsilon),
                                    float(d['modes'][name]['derivative'][i]), abs_tol=3e-8)
                loss = lambda da, db: .5 * (function(da, db) - 1) ** 2
                assert math.isclose(derivative(loss, (0, 0), i, epsilon),
                                    float(d['modes'][name]['gradient'][i]), abs_tol=5e-8)
                checks += 2
    for initial_a, initial_b, alpha, target in ((.5, 1., .5, 1.), (.3, .8, .2, .9), (-.2, 1.2, .1, -.3)):
        assert math.isclose(online(initial_a, initial_b, alpha, target)['h'],
                            learner_polynomial(initial_a, initial_b, alpha, target), abs_tol=1e-12)
    alpha_fd = derivative(lambda x: online(.5, 1., x)['h'], (.5,), 0)
    assert math.isclose(alpha_fd, float(d['alpha_derivative']), abs_tol=1e-9)
    # With alpha=0 held fixed, all three a,b derivative objects coincide.
    frozen = online(alpha=F(0))
    assert frozen['h'] == F(1, 8) and frozen['trace'] == (F(3, 4), F(1, 8))
    assert math.isclose(derivative(lambda a, b: online(a, b, 0.)['h'], (.5, 1.), 0), .75, abs_tol=1e-9)
    # Merely changing the current dictionary does not rewrite stored activity.
    assert actual['rows'][1]['h'] == F(1, 2)
    assert actual['rows'][1]['h'] != actual['current'][0] * actual['current'][1]
    assert d['modes']['schedule']['derivative'][0] != d['modes']['learner']['derivative'][0]
    print(f'PASS: {checks} state/loss finite differences, exact fractions, independent polynomial, alpha and zero-update boundary.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', action='store_true')
    if parser.parse_args().test:
        tests()
    else:
        print(json.dumps(results(), default=float, indent=2))
