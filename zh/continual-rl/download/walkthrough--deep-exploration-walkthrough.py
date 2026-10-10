#!/usr/bin/env python3
"""Standalone standard-library corridor tutorial and exact Fraction oracle.

python3 deep-exploration-walkthrough.py --test
python3 deep-exploration-walkthrough.py
The printed paths are specified legal quantile paths, not iid training runs.
"""
import argparse
from fractions import Fraction as F
from itertools import product
import json

LENGTH = 10
SAFE = F(1, 4)
MODEL = {'low': F(1, 5), 'high': F(4, 5)}


def belief(successes, failures, prior=F(1, 2)):
    """Batch likelihood, independently of the JavaScript one-observation filter."""
    high = prior * MODEL['high'] ** successes * (1 - MODEL['high']) ** failures
    low = (1 - prior) * MODEL['low'] ** successes * (1 - MODEL['low']) ** failures
    return high / (high + low)


def enumerate_directions(length, horizon):
    count = 0
    for actions in product((-1, 1), repeat=horizon):
        x = 0
        for a in actions:
            x = min(length, max(0, x + a)) if x < length else x
        count += x == length
    return F(count, 2 ** horizon)


def sample_clock_outcomes(q, mode, length):
    """List disjoint observable trial endings, including their physical cost."""
    if mode == 'held':
        return [(1-q, None, 1), (q, 'endpoint', length)]
    if mode == 'step':
        return [(q ** (t-1) * (1-q), None, t) for t in range(1, length+1)] + [(q ** length, 'endpoint', length)]
    raise ValueError('unknown sampling clock')


def enumerate_trials(episodes, theta, mode='held', length=LENGTH, prior=F(1, 2)):
    """Forward enumeration of probability mass and full histories, no DP value."""
    # mass, successes, failures, total reward, endpoint visits, physical actions
    frontier = [(F(1), 0, 0, F(0), 0, 0)]
    for _ in range(episodes):
        next_frontier = []
        for mass, successes, failures, reward, visits, actions in frontier:
            q = belief(successes, failures, prior)
            for branch_mass, ending, cost in sample_clock_outcomes(q, mode, length):
                if not branch_mass:
                    continue
                if ending is None:
                    next_frontier.append((mass*branch_mass, successes, failures, reward+SAFE, visits, actions+cost))
                else:
                    for y in (0, 1):
                        likelihood = MODEL[theta] if y else 1-MODEL[theta]
                        next_frontier.append((mass*branch_mass*likelihood, successes+y, failures+1-y, reward+y, visits+1, actions+cost))
        frontier = next_frontier
    assert sum(v[0] for v in frontier) == 1
    return {'value': sum(v[0]*v[3] for v in frontier),
            'endpoint_visits': sum(v[0]*v[4] for v in frontier),
            'primitive_actions': sum(v[0]*v[5] for v in frontier)}


def specified_trials(theta, quantiles=(F(2, 5), F(2, 5), F(1, 10)), reward_quantiles=(F(2, 5),)*3, prior=F(1, 2)):
    successes = failures = 0
    episodes = []
    for u, v in zip(quantiles, reward_quantiles):
        q = belief(successes, failures, prior)
        high = u < q
        records = []
        for t in range(1, LENGTH+1 if high else 2):
            endpoint = high and t == LENGTH
            y = int(v < MODEL[theta]) if endpoint else None
            reward = SAFE if not high else F(y or 0)
            after = belief(successes+(y or 0), failures+int(y == 0), prior) if endpoint else q
            records.append({'t': t, 'x': t-1 if high else 0, 'remaining': LENGTH-t+1,
                            'sampled': 'high' if high else 'low', 'action': 'right' if high else 'safe',
                            'next': t if high else 0, 'reward': reward,
                            'terminal': endpoint or not high, 'endpoint_observation': y,
                            'q_before': q, 'q_after': after})
        if high:
            successes += records[-1]['endpoint_observation']
            failures += 1-records[-1]['endpoint_observation']
        episodes.append({'mode': 'held', 'q_before': q, 'q_after': belief(successes, failures, prior),
                         'total': sum(r['reward'] for r in records), 'end': 'endpoint' if high else 'safe',
                         'primitive_actions': len(records), 'records': records})
    return {'episodes': episodes, 'q_final': belief(successes, failures, prior),
            'total': sum(e['total'] for e in episodes), 'endpoint_visits': sum(e['end'] == 'endpoint' for e in episodes)}


def output():
    return {'curves': [{'length': h, 'step_reach': enumerate_directions(h, h),
                        'held_reach': F(1, 2), 'model_step_reach': F(1, 2)**h} for h in range(1, 11)],
            'closed_loop': {'failure': specified_trials('low'),
                            'success': specified_trials('high', (F(2, 5),)*2, (F(1, 10), F(9, 10)))},
            'exact': {theta: {mode: enumerate_trials(3, theta, mode) for mode in ('held', 'step')} for theta in MODEL},
            'known_high_failure': specified_trials('high', (F(2, 5),), (F(9, 10),), F(1))}


def checks():
    assert enumerate_directions(10, 10) == F(1, 1024)
    assert belief(0, 1) == F(1, 5)
    assert belief(1, 0) == F(4, 5)
    assert belief(1, 1) == F(1, 2)
    d = output()
    failure = d['closed_loop']['failure']['episodes']
    assert [e['end'] for e in failure] == ['endpoint', 'safe', 'endpoint']
    assert [e['primitive_actions'] for e in failure] == [10, 1, 10]
    assert failure[0]['q_after'] == F(1, 5) and failure[1]['q_after'] == F(1, 5)
    assert d['known_high_failure']['episodes'][0]['q_after'] == 1
    for theta in MODEL:
        for mode in ('held', 'step'):
            x = enumerate_trials(1, theta, mode)
            reach = F(1, 2) if mode == 'held' else F(1, 1024)
            assert x['endpoint_visits'] == reach
            assert x['value'] == (1-reach)*SAFE + reach*MODEL[theta]
            assert x['primitive_actions'] == (F(11, 2) if mode == 'held' else F(1023, 512))
    # Independent complete model-draw strings match the outcome partition.
    mass = sum(F(1, 2)**10 for choices in product((0, 1), repeat=10) if all(choices))
    assert mass == F(1, 1024)
    print('Exact Fraction and specified action-record checks passed.')


def json_number(x):
    if isinstance(x, F):
        return float(x)
    raise TypeError(type(x).__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    if args.test:
        checks()
    else:
        print(json.dumps(output(), default=json_number, ensure_ascii=False, indent=2))
