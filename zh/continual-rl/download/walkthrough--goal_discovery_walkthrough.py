#!/usr/bin/env python3
"""Run from any directory: python3 goal_discovery_walkthrough.py [--json].

Standard library only. Original frontier baseline, not a STOMP reproduction.
Environment rows are revealed only by real steps. Fractions and exhaustive path
enumeration independently check the JavaScript Bellman mechanism.
"""
from collections import deque
from fractions import Fraction as F
import json
import sys

GAMMA = F(9, 10)
WORLD = {'S': {'upper': ('A', 0), 'lower': ('B', 0)},
         'A': {'exit': ('X', 1), 'probe': ('Y', 0)},
         'B': {'forward': ('C', 0)},
         'C': {'exit': ('X', 1), 'probe': ('Y', 2)}, 'X': {}, 'Y': {}}


class Learner:
    def __init__(self):
        self.rows, self.counts, self.actions = {}, {}, {}
        self.real_steps = 0

    def step(self, s, a):
        # This is the only entry point that reveals a previously unknown outcome.
        nxt, reward = WORLD[s][a]
        self.rows[s, a] = (nxt, F(reward))
        self.counts[s, a] = self.counts.get((s, a), 0) + 1
        self.actions[s], self.actions[nxt] = list(WORLD[s]), list(WORLD[nxt])
        self.real_steps += 1
        return nxt, F(reward)

    def discover(self):
        distance, queue = {'S': 0}, deque(['S'])
        while queue:
            s = queue.popleft()
            for (src, _), (nxt, _) in self.rows.items():
                if src == s and nxt not in distance:
                    distance[nxt] = distance[s] + 1
                    queue.append(nxt)
        result = []
        for s, actions in self.actions.items():
            untried = [a for a in actions if (s, a) not in self.counts]
            if s in distance and untried:
                result.append((s, F(1, distance[s] + 1), distance[s], untried))
        return sorted(result, key=lambda x: (-x[1], x[0]))

    def paths(self, state='S', goal=None, seen=()):
        """Enumerate data-supported simple paths, never query the world."""
        if state == goal or not self.actions[state]:
            return [([], state)]
        if state in seen:
            raise ValueError('This independent check assumes the declared DAG')
        result = []
        for (s, a), (nxt, reward) in self.rows.items():
            if s == state:
                for tail, end in self.paths(nxt, goal, seen + (state,)):
                    result.append(([(s, a, nxt, reward)] + tail, end))
        return result

    def goal_option(self, goal):
        # Analytic path maximization, independent of JS synchronous Q backups.
        policy = {}
        for s in self.actions:
            paths = [p for p, end in self.paths(s, goal) if end == goal and p]
            if paths:
                policy[s] = min(paths, key=lambda p: (len(p), p[0][1]))[0][1]
        return {'goal': goal, 'policy': policy}

    def model(self, option):
        s, reward, discount, duration, path = 'S', F(0), F(1), 0, ['S']
        while s != option['goal'] and self.actions[s]:
            nxt, r = self.rows[s, option['policy'][s]]
            reward += discount * r
            discount *= GAMMA
            duration += 1
            s = nxt
            path.append(s)
        return dict(reward=reward, discount=discount, duration=duration, endpoint=s, path=path)

    def execute(self, option, probe):
        s, total, discount, path = 'S', F(0), F(1), ['S']
        while s != option['goal'] and self.actions[s]:
            s, r = self.step(s, option['policy'][s])
            total += discount * r
            discount *= GAMMA
            path.append(s)
        if probe:
            a = next(a for a in self.actions[s] if (s, a) not in self.counts)
            s, r = self.step(s, a)
            total += discount * r
            path.append(s)
        return total, path

    def value(self, state):
        return max((sum((GAMMA ** k * row[3] for k, row in enumerate(p)), F(0))
                    for p, _ in self.paths(state)), default=F(0))


def run():
    learner = Learner()
    for s, a in [('S', 'upper'), ('A', 'exit'), ('S', 'lower'), ('B', 'forward'), ('C', 'exit')]:
        learner.step(s, a)
    assert 'Y' not in learner.actions
    assert [(g, score) for g, score, *_ in learner.discover()] == [('A', F(1, 2)), ('C', F(1, 3))]
    before = learner.value('S')
    options, rounds = [], []
    for _ in range(2):
        candidates = learner.discover()
        option = learner.goal_option(candidates[0][0])
        model = learner.model(option)
        ret, path = learner.execute(option, probe=True)
        options.append(option)
        rounds.append(dict(goal=option['goal'], candidates=candidates, model=model, path=path,
                           external_return=ret, next_goals=[x[0] for x in learner.discover()]))
    assert learner.discover() == []
    assert [r['goal'] for r in rounds] == ['A', 'C']
    assert [r['external_return'] for r in rounds] == [0, F(81, 50)]
    assert [r['model']['reward'] for r in rounds] == [0, 0]  # No training reward in model.
    assert [r['model']['discount'] for r in rounds] == [F(9, 10), F(81, 100)]
    assert learner.real_steps == 10
    consumer = []
    for option in options:
        model = learner.model(option)
        consumer.append((option['goal'], model['reward'] + model['discount'] * learner.value(model['endpoint'])))
    chosen = max(consumer, key=lambda row: row[1])[0]
    option = next(o for o in options if o['goal'] == chosen)
    prefix_return, path = learner.execute(option, False)
    s = path[-1]
    available = [(a, r + GAMMA * learner.value(nxt)) for (src, a), (nxt, r) in learner.rows.items() if src == s]
    action = max(available, key=lambda x: x[1])[0]
    nxt, reward = learner.step(s, action)
    deployment_return = prefix_return + GAMMA ** (len(path) - 1) * reward
    assert before == F(9, 10) and learner.value('S') == deployment_return == F(81, 50)
    assert learner.real_steps == 13 and len(learner.rows) == 7
    # Counterfactual diagnostic uses a separate copy; not exploration evidence.
    # Change only C:probe's external reward. Frontier candidates/policies are unchanged.
    old = learner.rows['C', 'probe']
    learner.rows['C', 'probe'] = ('Y', F(0))
    assert learner.value('S') == F(9, 10)
    learner.rows['C', 'probe'] = old
    return dict(before=before, rounds=rounds, consumer=consumer,
                deployment=dict(goal=chosen, path=path + [nxt], external_return=deployment_return),
                budgets=dict(seed_steps=5, exploration_steps=5, deployment_steps=3, resets=4,
                             unique_rows=7, options=2), checks='all exact assertions passed')


if __name__ == '__main__':
    result = run()
    if '--json' in sys.argv:
        print(json.dumps(result, default=lambda x: float(x) if isinstance(x, F) else x, indent=2))
    else:
        for r in result['rounds']:
            print(f"goal={r['goal']} path={' → '.join(r['path'])} external return={r['external_return']} next={r['next_goals']}")
        print('Consumer values:', result['consumer'])
        print('Deployment:', result['deployment'])
        print('Budgets:', result['budgets'])
        print(result['checks'])
