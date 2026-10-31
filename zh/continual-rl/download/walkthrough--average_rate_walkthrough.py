#!/usr/bin/env python3
"""Two service cycles, one physical clock. Standard library; no random training.

Download this file as average_rate_walkthrough.py, then run in its directory:
     python3 average_rate_walkthrough.py
     python3 average_rate_walkthrough.py --json

This independent implementation solves all Poisson equations using exact rational
Gaussian elimination. The JS figure kernel instead uses regenerative cycles.
Rewards arrive only when an event completes; E is visited once on startup.
"""
from fractions import Fraction as F
import json
import sys

STATES = ('E', 'H', 'A', 'B')


def model(slow_return=3):
    if not isinstance(slow_return, int) or slow_return < 1:
        raise ValueError('positive integer duration required')
    return {
        'E': [('enter', 'H', F(-3), F(2))],
        'H': [('quick', 'A', F(0), F(1)), ('slow', 'B', F(0), F(1))],
        'A': [('return', 'H', F(4), F(1))],
        'B': [('return', 'H', F(9), F(slow_return))],
    }


def solve(matrix, rhs):
    """Exact elimination, including the explicitly supplied gauge equation."""
    a = [[F(x) for x in row]+[F(b)] for row, b in zip(matrix, rhs)]
    n = len(a)
    for j in range(n):
        pivot = next(i for i in range(j, n) if a[i][j])
        a[j], a[pivot] = a[pivot], a[j]
        scale = a[j][j]
        a[j] = [v/scale for v in a[j]]
        for i in range(n):
            if i != j:
                coefficient = a[i][j]
                a[i] = [x-coefficient*y for x, y in zip(a[i], a[j])]
    return [row[-1] for row in a]


def evaluate(slow_probability=F(0), slow_return=3):
    p = F(slow_probability)
    if not 0 <= p <= 1:
        raise ValueError('probability in [0,1] required')
    matrix, rhs = [], []
    for i, s in enumerate(STATES):
        row = [F(0)]*5  # unknowns h(E), h(H), h(A), h(B), g
        row[i] = 1
        reward = F(0)
        for a, next_state, r, duration in model(slow_return)[s]:
            probability = ((1-p) if a == 'quick' else p) if s == 'H' else F(1)
            row[STATES.index(next_state)] -= probability
            row[4] += probability*duration
            reward += probability*r
        matrix.append(row)
        rhs.append(reward)
    matrix.append([0, 1, 0, 0, 0])  # h(H)=0, not event-stationary centering
    rhs.append(0)
    solution = solve(matrix, rhs)
    return {'gain': solution[4], 'bias': dict(zip(STATES, solution[:4]))}


def action_values(evaluation, slow_return=3):
    g, h = evaluation['gain'], evaluation['bias']
    return {a: r-g*duration+h[n] for a, n, r, duration in model(slow_return)['H']}


def duration_backup(reward, duration, rate, current, next_value, alpha=F(3, 10), eta=F(1, 2)):
    """One inter-option backup with known deterministic expected duration.

    delta is computed once from OLD values. This is a single arithmetic step,
    not a convergence test or an estimator that can learn transient E forever.
    """
    if duration <= 0:
        raise ValueError('positive expected duration required')
    delta = reward-rate*duration+next_value-current
    increment = alpha*delta/duration
    return {'delta': delta, 'increment': increment,
            'value': current+increment, 'rate': rate+eta*increment}


def lifetime(policy='quick', horizon=18, slow_return=3):
    if policy not in ('quick', 'slow') or not isinstance(horizon, int) or horizon < 0:
        raise ValueError('policy and integer horizon required')
    transitions = model(slow_return)
    state, total, remaining, pending = 'E', F(0), 0, None
    points = [[0, F(0)]]
    # Expand every event into one-second ticks, including incomplete final events.
    for second in range(1, horizon+1):
        if remaining == 0:
            pending = next(e for e in transitions[state] if state != 'H' or e[0] == policy)
            remaining = int(pending[3])
        remaining -= 1
        if remaining == 0:
            total += pending[2]
            state = pending[1]
        points.append([second, total])
    return {'points': points, 'physicalSeconds': horizon, 'totalReward': total}


def results():
    quick, slow = evaluate(), evaluate(F(1))
    return {'quick': quick, 'slow': slow, 'mixed': evaluate(F(1, 2)),
            'quickActionValues': action_values(quick), 'slowActionValues': action_values(slow),
            'changedQuick': evaluate(F(0), 4), 'changedSlow': evaluate(F(1), 4),
            'changedQuickActionValues': action_values(evaluate(F(0), 4), 4),
            'predictionBackup': duration_backup(F(9), F(3), F(2), F(2), F(0)),
            'quickLifetime': lifetime(), 'slowLifetime': lifetime('slow'),
            'changedSlowLifetime': lifetime('slow', 18, 4)}


if __name__ == '__main__':
    data = results()
    if '--json' in sys.argv:
        print(json.dumps(data, default=float))
    else:
        for name in ('quick', 'slow', 'mixed'):
            print(name, 'gain =', data[name]['gain'], 'bias =', data[name]['bias'])
        print('quick-policy action values:', data['quickActionValues'])
        print('slow-policy action values:', data['slowActionValues'])
        print('one old-snapshot backup:', data['predictionBackup'])
        for t in (12, 18):
            print('reward at', t, 'seconds:', lifetime('quick', t)['totalReward'], lifetime('slow', t)['totalReward'])
        print('changed slow gain:', data['changedSlow']['gain'])
        print('corrected model action values:', data['changedQuickActionValues'])
