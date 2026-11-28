#!/usr/bin/env python3
"""Two corridor episodes, exactly, using Python 3 standard library only.

Save this one file in any directory and run:
    python3 feature-control-walkthrough.py
    python3 feature-control-walkthrough.py --test

The environment returns A continue -> (2,B), A exit -> (0,T), and
B continue -> (-1,T). The learner sees only the current transition.
Truth below is a reader's oracle, never an update target.

Independent oracle: update predictions in Gram space using rational feature
overlap, rather than reproducing the website's floating weight-vector writes.
Normalized tile entries are 1/sqrt(2); their inner product is intersection/2.
The value 1/2 is the inner product, NOT a normalized feature component.
"""
from fractions import Fraction as F
import json
import sys
import unittest

TASK = {'A': [(2, 'B'), (0, 'T')], 'B': [(-1, 'T')]}
KEYS = {'separate': {'A': {0}, 'B': {1}},
        'tiles': {'A': {(0, 0), (1, 1)}, 'B': {(0, 1), (1, 1)}},
        'aggregate': {'A': {0}, 'B': {0}}}


def gram(name, scale=F(1)):
    keys = KEYS[name]
    count = len(keys['A'])
    return {s: {u: F(len(keys[s] & keys[u]), count) * scale**2
                for u in keys} for s in keys}


def run(name, scale=F(1), alpha=F(3, 5), episodes=2):
    """Control reads estimates, then environment supplies a single experience."""
    kernel = gram(name, scale)
    q = {'A': [F(0), F(0)], 'B': [F(0), F(0)]}
    events, first = [], None
    for episode in range(1, episodes + 1):
        state, action = 'A', int(q['A'][1] > q['A'][0])
        while state != 'T':
            before = {'A': q['A'][:], 'B': q['B'][:1]}
            reward, next_state = TASK[state][action]
            terminal = next_state == 'T'
            # B has one available action; T has none. Retain this choice.
            next_action = None if terminal else 0
            bootstrap = F(0) if terminal else q[next_state][next_action]
            target, old_q = F(reward) + bootstrap, q[state][action]
            delta = target - old_q
            # Exact prediction-space update, independent of weight coordinates.
            for probe in q:
                q[probe][action] += alpha * delta * kernel[probe][state]
            after = {'A': q['A'][:], 'B': q['B'][:1]}
            events.append(dict(episode=episode, state=state, action=action,
                               reward=reward, next=next_state, terminated=terminal,
                               next_action=next_action, before=before, old_q=old_q,
                               bootstrap=bootstrap, target=target, delta=delta, after=after))
            state, action = next_state, next_action
        if episode == 1:
            first = {'A': q['A'][:], 'B': q['B'][:1]}
    return dict(events=events, after_first_episode=first,
                next_real_transition=next(e for e in events if e['episode'] == 2),
                final={'A': q['A'][:], 'B': q['B'][:1]})


def complete_return(state, action):
    """Independent finite path enumeration, used by the reader only."""
    reward, following = TASK[state][action]
    rewards = [reward]
    while following != 'T':
        reward, following = TASK[following][0]
        rewards.append(reward)
    return sum(rewards)


def data():
    reps = {name: run(name) for name in KEYS}
    return dict(analysis_truth={'A': [complete_return('A', a) for a in [0, 1]],
                               'B': [complete_return('B', 0)]},
                representations=reps,
                scales={'base': reps['tiles'], 'uncompensated': run('tiles', F(10)),
                        'compensated': run('tiles', F(10), F(3, 500))})


def numbers(obj):
    if isinstance(obj, F):
        return float(obj)
    if isinstance(obj, dict):
        return {k: numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [numbers(v) for v in obj]
    return obj


class Checks(unittest.TestCase):
    def test_truth_by_paths(self):
        self.assertEqual([complete_return('A', a) for a in [0, 1]], [1, 0])
        self.assertEqual(complete_return('B', 0), -1)

    def test_raw_tiling_not_same_number_same_feature(self):
        # An independent implementation of floor(s/h + k/K).
        keys = lambda s: {(k, (s / F(1, 2) + F(k, 2)) // 1) for k in range(2)}
        self.assertEqual(keys(F(3, 10)) & keys(F(3, 5)), {(1, 1)})

    def test_gram(self):
        self.assertEqual([gram(n)['A']['B'] for n in KEYS], [0, F(1, 2), 1])
        self.assertTrue(all(gram(n)['A']['A'] == 1 for n in KEYS))

    def test_first_step_before_future_reward(self):
        for name in KEYS:
            event = run(name)['events'][0]
            self.assertEqual(event['target'], 2)
            self.assertEqual(event['after']['A'][0], F(6, 5))

    def test_first_episode(self):
        self.assertEqual([run(n)['after_first_episode']['A'][0] for n in KEYS],
                         [F(6, 5), F(18, 25), F(-3, 25)])
        self.assertEqual([run(n)['after_first_episode']['B'][0] for n in KEYS],
                         [F(-3, 5), F(-9, 25), F(-3, 25)])

    def test_actions_change_experience(self):
        for name in KEYS:
            event = run(name)['next_real_transition']
            self.assertEqual((event['reward'], event['next']),
                             (0, 'T') if name == 'aggregate' else (2, 'B'))

    def test_terminal_is_reward_only(self):
        self.assertTrue(all(run(n)['events'][1]['target'] == -1 for n in KEYS))
        self.assertTrue(all(run(n)['events'][1]['next_action'] is None for n in KEYS))

    def test_scale_without_compensation(self):
        result = run('tiles', F(10))
        self.assertEqual(result['events'][0]['after']['A'][0], 120)
        self.assertEqual(result['after_first_episode']['A'][0], -1710)
        self.assertEqual(result['after_first_episode']['B'][0], -3600)
        self.assertEqual(result['next_real_transition']['action'], 1)

    def test_scale_compensation_every_event(self):
        self.assertEqual(run('tiles'), run('tiles', F(10), F(3, 500)))

    def test_binary_equivalence(self):
        # Raw two-tile features have norm^2=2: alpha=.3 matches normalized alpha=.6.
        norm_squared, alpha_raw = F(2), F(3, 10)
        overlap = F(1)
        a1 = alpha_raw * 2 * norm_squared
        b1 = alpha_raw * 2 * overlap
        a2 = a1 + alpha_raw * (-1 - b1) * overlap
        self.assertEqual(a2, run('tiles')['after_first_episode']['A'][0])


if __name__ == '__main__':
    if '--test' in sys.argv:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
        result = unittest.TextTestRunner().run(suite)
        if not result.wasSuccessful():
            sys.exit(1)
        print('PASS: 10 independent Fraction checks')
    else:
        print(json.dumps(numbers(data()), indent=2))
