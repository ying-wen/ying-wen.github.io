#!/usr/bin/env python3
"""Two real episodic door interactions; no dependencies, training sweep, or hidden labels.

Run: python3 recurrent-control-walkthrough.py [--json | --test]
The Fraction oracle expands the decision formula independently of the JS recurrence.
The environment alone owns z. Each episode has two transitions and one reset call.
"""
import argparse
import json
from fractions import Fraction as F


class Door:
    def __init__(self, z, cue):
        if z not in (-1, 1) or cue not in (-1, 1):
            raise ValueError('signed bits required')
        self.__z = z
        self.place = 'C'
        self.initial = dict(place='C', signal=cue, reward=0, terminal=False)

    def step(self, action):
        if self.place == 'C' and action == 'forward':
            self.place = 'J'
            return dict(place='J', signal=0, reward=0, terminal=False)
        if self.place != 'J' or action not in (-1, 1):
            raise ValueError('invalid or post-terminal action')
        self.place = 'L' if action == 1 else 'R'
        reward = action * self.__z
        return dict(place=self.place, signal=reward, reward=reward, terminal=True)


def episode(env, a, alpha=F(1, 2), recorded_action=None, erase_cue=False):
    # Independent closed-form calculation for the two pre-decision observations.
    cue = F(0 if erase_cue else env.initial['signal'])
    arrival = env.step('forward')
    h_j = a * cue + arrival['signal']
    s_j = cue  # d(a*c + 0)/da; initial memory is zero.
    selected = 1 if h_j >= 0 else -1
    action = selected if recorded_action is None else recorded_action
    q = action * h_j
    dq = action * s_j
    outcome = env.step(action)
    reward = outcome['reward']
    gradient = (q - reward) * dq
    new_a = a - alpha * gradient
    return dict(usedParameter=a, decisionState=h_j, decisionSensitivity=s_j,
                selectedAction=selected, action=action, prediction=q, derivative=dq,
                reward=reward, gradient=gradient, updatedParameter=new_a,
                terminalState=a * a * cue + outcome['signal'],
                terminalSensitivity=2 * a * cue, environmentSteps=2, resetCalls=1,
                terminated=True, truncated=False)


def run(draws, a=F(-1, 2), **options):
    episodes = []
    for z, cue in draws:
        row = episode(Door(z, cue), a, **options)
        episodes.append(row)
        a = row['updatedParameter']
    return dict(episodes=episodes, finalParameter=a,
                totalReward=sum(e['reward'] for e in episodes),
                environmentSteps=2 * len(episodes), resetCalls=len(episodes))


def enumerate_two(accuracy=F(4, 5), **options):
    cases = [(z, c, F(1, 2) * (accuracy if z == c else 1 - accuracy))
             for z in (-1, 1) for c in (-1, 1)]
    mass, r1, r2, success = F(0), F(0), F(0), F(0)
    for z1, c1, p1 in cases:
        for z2, c2, p2 in cases:
            first, second = run([(z1, c1), (z2, c2)], **options)['episodes']
            p = p1 * p2
            mass += p
            r1 += p * first['reward']
            r2 += p * second['reward']
            success += p * (second['reward'] == 1)
    return dict(probability=mass, expectedRewards=[r1, r2], secondSuccess=success,
                expectedTotal=r1 + r2)


def recorded_training(draws):
    records = []
    for z, cue in draws:
        env = Door(z, cue)
        records.append((env.initial, env.step('forward'), env.step(-1)))

    class Record:
        def __init__(self, record):
            self.initial, self.junction, self.terminal = record
        def step(self, action):
            if action == 'forward':
                return self.junction
            if action != -1:
                raise ValueError('requested action differs from fixed right-door record')
            return self.terminal

    a, episodes = F(-1, 2), []
    for record in records:
        row = episode(Record(record), a, recorded_action=-1)
        row.update(environmentSteps=0, resetCalls=0, recordedTransitions=2)
        episodes.append(row)
        a = row['updatedParameter']
    return dict(episodes=episodes, finalParameter=a,
                totalReward=sum(e['reward'] for e in episodes), environmentSteps=0,
                resetCalls=0, recordedTransitions=2*len(records),
                collectionEnvironmentSteps=2*len(records), collectionResetCalls=len(records))


def data():
    selected = [(1, 1), (1, 1)]
    return dict(live=run(selected), record=recorded_training(selected),
                erased=run(selected, erase_cue=True), frozen=run(selected, alpha=F(0)),
                misleading=run([(-1, 1), (1, 1)]), enumeration=enumerate_two(),
                frozenEnumeration=enumerate_two(alpha=F(0)),
                noCueEnumeration=enumerate_two(erase_cue=True),
                noInformationEnumeration=enumerate_two(accuracy=F(1, 2)))


def tests():
    d = data()
    first, second = d['live']['episodes']
    assert [e['action'] for e in d['live']['episodes']] == [-1, 1]
    assert [e['reward'] for e in d['live']['episodes']] == [-1, 1]
    assert [e['updatedParameter'] for e in d['live']['episodes']] == [F(1, 4), F(5, 8)]
    assert first['terminalState'] == F(-3, 4)
    assert second['terminalState'] == F(17, 16)
    assert d['record']['episodes'][1]['terminalState'] == F(-15, 16)
    assert d['record']['finalParameter'] == d['live']['finalParameter']
    assert d['record']['totalReward'] == -2 and d['live']['totalReward'] == 0
    assert d['enumeration']['probability'] == 1
    assert d['enumeration']['expectedRewards'] == [F(-3, 5), F(9, 25)]
    assert d['enumeration']['secondSuccess'] == F(17, 25)
    assert d['enumeration']['expectedTotal'] == F(-6, 25)
    assert d['frozenEnumeration']['expectedTotal'] == F(-6, 5)
    assert d['noCueEnumeration']['expectedRewards'] == [0, 0]
    assert d['noInformationEnumeration']['expectedRewards'] == [0, 0]
    assert all(e['gradient'] == 0 for e in d['erased']['episodes'])
    assert d['misleading']['episodes'][0]['updatedParameter'] == F(-3, 4)
    # Exact reward-frequency calculation independent of enumerating latent draws.
    p = F(4, 5)
    assert p*p + (1-p)*(1-p) == d['enumeration']['secondSuccess']
    # A held-action, held-outcome loss. This does not differentiate the argmax or environment.
    for a in (F(-1, 2), F(1, 4), F(3, 4)):
        for cue in (-1, 1):
            for action in (-1, 1):
                for reward in (-1, 1):
                    eps = F(1, 10000)
                    loss = lambda x: (action*x*cue - reward)**2 / 2
                    fd = (loss(a+eps)-loss(a-eps))/(2*eps)
                    assert fd == (action*a*cue-reward)*action*cue
    # The antisymmetric reward/head makes the write independent of the recorded door.
    for z in (-1, 1):
        for cue in (-1, 1):
            left = episode(Door(z, cue), F(1, 4), recorded_action=1)
            right = episode(Door(z, cue), F(1, 4), recorded_action=-1)
            assert left['updatedParameter'] == right['updatedParameter']
            assert left['updatedParameter'] == F(1, 8) + F(1, 2)*z*cue
    env = Door(1, 1)
    episode(env, F(-1, 2))
    try:
        env.step(1)
        raise AssertionError('post-terminal action accepted')
    except ValueError:
        pass
    print('PASS: exact writes, 16-path enumeration, finite differences, data divergence, and degeneracies')


def serial(value):
    if isinstance(value, F):
        return float(value)
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serial(v) for v in value]
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true', help='print all exact-enumeration quantities as JSON numbers')
    parser.add_argument('--test', action='store_true', help='run standard-library exact checks')
    args = parser.parse_args()
    if args.test:
        tests()
    elif args.json:
        print(json.dumps(serial(data()), indent=2))
    else:
        d = data()
        for mode in ('live', 'record'):
            print(mode)
            for i, row in enumerate(d[mode]['episodes'], 1):
                print(' episode', i, 'a =', row['usedParameter'], 'door =', row['action'],
                      'R =', row['reward'], 'h_terminal =', row['terminalState'],
                      'a_next =', row['updatedParameter'])
        print('Exact 16-path reward expectations:', d['enumeration']['expectedRewards'])
        print('4 environment transitions and 2 reset calls per two-episode run; no budget truncation.')
