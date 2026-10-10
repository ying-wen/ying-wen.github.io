"""Goal/model/planning: standard-library exact rational reference, no training RNG.

Run: python3 goal_model_planning_walkthrough.py
All observations are a prescribed deterministic fixture. The planner reads only
observations. Known current dynamics are used solely for read-only diagnostics.
"""
from fractions import Fraction as F
import json

GAMMA = F(9, 10)
ACTIONS = {'S': ('upper', 'lower'), 'A': ('cross',), 'B': ('forward',),
           'C': ('forward',), 'X': (), 'Y': ()}
HISTORY = [('S', 'upper', 'A', False), ('A', 'cross', 'X', True),
           ('S', 'lower', 'B', False), ('B', 'forward', 'C', False),
           ('C', 'forward', 'X', True)]
NEW = [('S', 'upper', 'A', False), ('A', 'cross', 'Y', True)]


def learn(records, initial=None):
    model = dict(initial or {})
    for s, a, nxt, physical_done in records:
        assert a in ACTIONS[s] and nxt in ACTIONS
        model[(s, a)] = (nxt, physical_done)
    return model


def label(s, nxt, physical_done, goal):
    if s == goal or not ACTIONS[s]:
        raise ValueError('No action after task termination')
    return F(nxt == goal), physical_done or nxt == goal


def solve(model, goal, rounds=4, initial=None):
    q = dict(initial) if initial is not None else {(s, a): F(0) for s in ACTIONS for a in ACTIONS[s]}
    rows, calls = [], 0
    for k in range(rounds + 1):
        rows.append({'k': k, 'calls': calls, 'q': {f'{s}:{a}': float(v) for (s, a), v in q.items()},
                     'choice': 'upper' if q[('S', 'upper')] > q[('S', 'lower')] else 'lower'})
        if k == rounds:
            break
        new_q = {}
        for s, a in q:
            if s == goal:
                new_q[(s, a)] = F(0)
                continue
            nxt, physical_done = model[(s, a)]
            calls += 1
            reward, done = label(s, nxt, physical_done, goal)
            tail = max((q[(nxt, b)] for b in ACTIONS[nxt]), default=F(0))
            new_q[(s, a)] = reward + (F(0) if done else GAMMA * tail)
        q = new_q
    return rows, q


def main():
    old = learn(HISTORY)
    updated = learn(NEW, old)
    x, qx = solve(old, 'X')
    a, _ = solve(old, 'A')
    repaired, _ = solve(updated, 'X', 3, qx)
    stale, _ = solve(old, 'X', 3, qx)
    assert qx[('S', 'upper')] == F(9, 10)
    assert qx[('S', 'lower')] == F(81, 100)
    assert repaired[1]['choice'] == 'upper' and repaired[2]['choice'] == 'lower'
    assert label('S', 'A', False, 'A') == (F(1), True)
    assert label('A', 'Y', True, 'X') == (F(0), True)
    print(json.dumps({'goals': {'X': x, 'A': a}, 'afterChange': {'repaired': repaired, 'stale': stale},
                      'environmentSteps': {'old': 5, 'new': 2, 'total': 7}, 'resets': 2}, indent=2))


if __name__ == '__main__':
    main()
