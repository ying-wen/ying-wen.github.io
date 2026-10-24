"""Two planning calls change the next real data: a finite Dyna walkthrough.

Python 3.10+, standard library only. Run from any working directory:
    python3 /path/to/architecture-walkthrough.py
The output contains every real observation, model call, Q update and counter.
Q and the model start empty/zero. Four prescribed real actions warm them up;
the next three actions are selected by the learner. No random training occurs.
The environment owns outcomes; the planner only reads observed model rows.
"""
from fractions import Fraction
import json

ACTIONS = {'S': ('near', 'far'), 'A': ('go',), 'B': ('go',), 'D': (), 'G': ()}


class Environment:
    def __init__(self, near_reward=2, far_reward=6):
        self.state, self.steps, self.resets = 'S', 0, 0
        self._outcomes = {('S', 'near'): ('D', near_reward), ('S', 'far'): ('A', 0),
                          ('A', 'go'): ('B', 0), ('B', 'go'): ('G', far_reward)}

    def step(self, action):
        if action not in ACTIONS[self.state]:
            raise ValueError('Action unavailable in current state')
        state = self.state
        self.state, reward = self._outcomes[state, action]
        self.steps += 1
        return state, action, Fraction(reward), self.state, not ACTIONS[self.state]

    def reset(self):
        if ACTIONS[self.state]:
            raise ValueError('Reset requires task termination')
        self.state = 'S'
        self.resets += 1


class Learner:
    def __init__(self, gamma=Fraction(9, 10), alpha=Fraction(1)):
        if not 0 <= gamma <= 1 or not 0 <= alpha <= 1:
            raise ValueError('Invalid discount or step size')
        self.gamma, self.alpha = gamma, alpha
        self.q = {(s, a): Fraction(0) for s, aa in ACTIONS.items() for a in aa}
        self.model, self.counts = {}, {}
        self.t, self.k = 0, 0

    def choose(self, state):
        if not ACTIONS[state]:
            raise ValueError('No decision at a terminal state')
        return max(ACTIONS[state], key=lambda a: self.q[state, a])

    def update(self, sample):
        s, a, reward, nxt, done = sample
        if done != (not ACTIONS[nxt]):
            raise ValueError('Inconsistent task termination')
        old = self.q[s, a]
        tail = Fraction(0) if done else max(self.q[nxt, b] for b in ACTIONS[nxt])
        target = reward + self.gamma * tail
        self.q[s, a] += self.alpha * (target - old)
        return {'key': f'{s}:{a}', 'before': old, 'target': target, 'after': self.q[s, a]}

    def observe(self, sample):
        update = self.update(sample)
        key = sample[:2]
        self.model[key] = sample
        self.counts[key] = self.counts.get(key, 0) + 1
        self.t += 1
        return update

    def plan(self, key):
        if key not in self.model:
            raise ValueError('Unobserved model row')
        update = self.update(self.model[key])
        self.k += 1
        return update

    def snapshot(self):
        return {'t': self.t, 'k': self.k,
                'q': {f'{s}:{a}': value for (s, a), value in self.q.items()},
                'counts': {f'{s}:{a}': n for (s, a), n in self.counts.items()},
                'choice': self.choose('S')}


def record(sample, update, learner):
    s, a, reward, nxt, done = sample
    return dict(state=s, action=a, reward=reward, next=nxt, done=done,
                update=update, **learner.snapshot())


def run(order, budget=2, follow_steps=3, gamma=Fraction(9, 10),
        alpha=Fraction(1), near_reward=2, far_reward=6):
    if order not in ('forward', 'reverse') or budget < 0 or follow_steps < 0:
        raise ValueError('Invalid order or budget')
    env = Environment(near_reward, far_reward)
    learner = Learner(gamma, alpha)
    warmup = []
    for action in ('near', 'far', 'go', 'go'):
        sample = env.step(action)
        warmup.append(record(sample, learner.observe(sample), learner))
        if sample[-1]:
            env.reset()
    before = learner.snapshot()
    keys = [(r['state'], r['action']) for r in warmup if not r['done']]
    if order == 'reverse':
        keys.reverse()
    planning = []
    for i in range(budget):
        update = learner.plan(keys[i % len(keys)])
        planning.append(dict(**update, **learner.snapshot()))
    decision = learner.snapshot()
    follow = []
    for _ in range(follow_steps):
        sample = env.step(learner.choose(env.state))
        follow.append(record(sample, learner.observe(sample), learner))
        if sample[-1]:
            env.reset()
    return {'order': order, 'warmup': warmup, 'before': before, 'planning': planning,
            'decision': decision, 'follow': follow, 'final': learner.snapshot(),
            'worldSteps': env.steps, 'resets': env.resets,
            'followReward': sum(row['reward'] for row in follow)}


def main():
    data = {order: run(order) for order in ('forward', 'reverse')}
    print(json.dumps(data, default=lambda x: float(x) if isinstance(x, Fraction) else x, indent=2))


if __name__ == '__main__':
    main()
