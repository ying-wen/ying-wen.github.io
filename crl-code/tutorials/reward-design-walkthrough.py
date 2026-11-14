#!/usr/bin/env python3
"""One parcel, three ticks: independent exact path enumeration (stdlib only).

Download this file alone. From any directory run:
  python3 reward-design-walkthrough.py --test
  python3 reward-design-walkthrough.py --json
The task really ends on delivery or after action 3. This is not an arbitrary
training cutoff. State includes the clock. No training, RNG, or external files.
"""
import argparse
from fractions import Fraction as F
import json

GAMMA = F(9, 10)
MODES = ('true', 'proxy', 'shaping', 'bad-boundary', 'bad-discount')


def complete_paths():
    """Enumerate all legal trajectories before applying any reward transform."""
    def extend(states, actions):
        t, x = states[-1]
        if x == 'G' or t == 3:
            yield states, actions
            return
        successors = [('approach', 'A')] if x == 'S' else [('deliver', 'G'), ('wait', 'A')]
        for a, nxt in successors:
            yield from extend(states + [(t + 1, nxt)], actions + [a])
    return list(extend([(0, 'S')], []))


def phi(state, keep_timeout=False):
    t, x = state
    if x == 'G' or (t == 3 and not keep_timeout):
        return F(0)
    return F(1 if x == 'S' else 3)


def evaluate(states, actions, mode, gamma=GAMMA):
    raw = [F(a == 'deliver') for a in actions]
    transformed = []
    for t, reward in enumerate(raw):
        if mode == 'true':
            extra = F(0)
        elif mode == 'proxy':
            extra = F(3 if states[t + 1][1] == 'A' else 0)
        else:
            factor = F(1) if mode == 'bad-discount' else gamma
            extra = factor * phi(states[t + 1], mode == 'bad-boundary') - phi(states[t])
        transformed.append(reward + extra)
    total = lambda rewards: sum((gamma**t * r for t, r in enumerate(rewards)), F(0))
    return dict(actions=actions, rewards=raw, training=transformed,
                trueReturn=total(raw), trainReturn=total(transformed),
                delivered=states[-1][1] == 'G', steps=len(actions))


def calculate(gamma=GAMMA):
    all_paths = complete_paths()
    names = ('early', 'late', 'timeout')
    results = []
    for mode in MODES:
        paths = [dict(id=name, **evaluate(states, actions, mode, gamma))
                 for name, (states, actions) in zip(names, all_paths)]
        best = max(paths, key=lambda p: p['trainReturn'])
        results.append(dict(mode=mode, paths=paths, selected=best['id'], run=best))
    return dict(results=results)


def test():
    data = calculate()
    # Closed-form return expressions independently expose every changed term.
    g = GAMMA
    exact = {
        'true': [g, g*g, F(0)],
        'proxy': [3+g, 3+3*g+g*g, 3*(1+g+g*g)],
        'shaping': [g-1, g*g-1, F(-1)],
        'bad-boundary': [g-1, g*g-1, -1+3*g**3],
        'bad-discount': [g-1+3*(1-g), g*g-1+3*(1-g)*(1+g), -1+3*(1-g)*(1+g)],
    }
    assert [r['selected'] for r in data['results']] == ['early', 'timeout', 'early', 'timeout', 'late']
    for result in data['results']:
        assert [p['trainReturn'] for p in result['paths']] == exact[result['mode']]
    for states, actions in complete_paths():
        shaped = evaluate(states, actions, 'shaping')
        bad = evaluate(states, actions, 'bad-boundary')
        assert shaped['trainReturn'] == shaped['trueReturn'] - phi(states[0])
        assert bad['trainReturn'] == bad['trueReturn'] - phi(states[0]) + g**len(actions)*phi(states[-1], True)
    # At gamma=1 the missing-discount implementation becomes the correct one.
    one = calculate(F(1))['results']
    assert [p['trainReturn'] for p in one[2]['paths']] == [p['trainReturn'] for p in one[4]['paths']]
    # A collection cutoff at t=1 is not task termination: retain the true tail.
    # At A1, immediate delivery has V=1; shaped tail is 1 - Phi(A1) = -2.
    assert (g*3-1) + g*(1-3) == g-1
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    data = test()
    if args.json:
        print(json.dumps(data, default=lambda x: float(x) if isinstance(x, F) else str(x)))
    elif args.test:
        print('PASS: exhaustive Fraction paths, closed-form returns, boundary and discount checks.')
    else:
        print('Exact known-model decisions; one episode per row; no learning-speed experiment.')
        print('signal          selected  training-return  true-return  actions')
        for r in data['results']:
            p = r['run']
            print(f"{r['mode']:15} {r['selected']:9} {str(p['trainReturn']):16} {str(p['trueReturn']):12} {'/'.join(p['actions'])}")


if __name__ == '__main__':
    main()
