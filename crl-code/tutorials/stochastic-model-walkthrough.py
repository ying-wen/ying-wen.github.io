#!/usr/bin/env python3
"""Joint empirical model and Dyna backups, using Python standard library only.

This is an exact mechanism fixture, not random training. Four prescribed exploratory
crossings generate a possible D,G,G,G record. True probabilities belong to the
reader's exhaustive reference; observe/model/planning never receive them.
Each real row writes Q directly (alpha=1) and updates joint outcome counts.
Expectation expands every observed outcome. Sampling uses an explicit quantile.
Only the first cross target in a comparison reads the common frozen Q; the later
go backup reads its branch's updated cross value. All terminals have zero tail.
Download this file alone; run --test or --json from any empty directory.
"""
import argparse
import copy
import itertools
import json
from fractions import Fraction as F

GAMMA = F(9, 10)
ACTIONS = {'S': ('go', 'exit'), 'A': ('cross',), 'G': (), 'D': (), 'B': ()}
GO = {'s': 'S', 'a': 'go', 'next': 'A', 'reward': F(0), 'done': False}
EXIT = {'s': 'S', 'a': 'exit', 'next': 'B', 'reward': F(3, 5), 'done': True}
COUNTERS = ('environmentSteps', 'resets', 'modelWrites', 'directBackups',
            'modelQueries', 'outcomeEvaluations', 'planningBackups')


def crossing(next_state):
    return {'s': 'A', 'a': 'cross', 'next': next_state,
            'reward': F(next_state == 'G'), 'done': True}


def new_learner():
    return {'q': {'S:go': F(0), 'S:exit': F(0), 'A:cross': F(0)},
            'model': {}, 'counts': dict.fromkeys(COUNTERS, 0)}


def row_target(q, row, gamma=GAMMA):
    if row['a'] not in ACTIONS.get(row['s'], ()) or row['next'] not in ACTIONS:
        raise ValueError('Invalid outcome')
    if row['done']:
        return row['reward']
    if not ACTIONS[row['next']]:
        raise ValueError('Terminal outcome must be marked done')
    return row['reward'] + gamma * max(q[f"{row['next']}:{a}"] for a in ACTIONS[row['next']])


def observe(l, row):
    y = row_target(l['q'], row)
    key = f"{row['s']}:{row['a']}"
    before = l['q'][key]
    l['q'][key] = y
    for counter in ('environmentSteps', 'directBackups', 'modelWrites'):
        l['counts'][counter] += 1
    entry = l['model'].setdefault(key, {'n': 0, 'outcomes': []})
    joint = (row['next'], row['reward'], row['done'])
    item = next((o for o in entry['outcomes']
                 if (o['next'], o['reward'], o['done']) == joint), None)
    if item is None:
        item = {'next': joint[0], 'reward': joint[1], 'done': joint[2], 'count': 0}
        entry['outcomes'].append(item)
    item['count'] += 1
    entry['n'] += 1
    entry['outcomes'].sort(key=lambda o: (o['next'], o['reward'], o['done']))
    return {'pair': key, 'before': before, 'target': y, 'after': y}


def distribution(l, key):
    if key not in l['model']:
        raise ValueError('Unobserved pair ' + key)
    entry = l['model'][key]
    return [dict(o, probability=F(o['count'], entry['n'])) for o in entry['outcomes']]


def model_target(l, key, mode='expected', u=F(1, 2), gamma=GAMMA):
    rows = distribution(l, key)
    s, a = key.split(':')
    l['counts']['modelQueries'] += 1
    if mode == 'expected':
        l['counts']['outcomeEvaluations'] += len(rows)
        return sum(o['probability'] * row_target(l['q'], dict(o, s=s, a=a), gamma) for o in rows)
    if mode != 'sample' or not F(0) <= u < F(1):
        raise ValueError('Invalid model query')
    mass = F(0)
    for row in rows:
        mass += row['probability']
        if u < mass:
            l['counts']['outcomeEvaluations'] += 1
            return row_target(l['q'], dict(row, s=s, a=a), gamma)
    raise AssertionError('Distribution not normalized')


def planning_backup(l, key, mode='expected', u=F(1, 2), alpha=F(1), gamma=GAMMA):
    y = model_target(l, key, mode, u, gamma)
    before = l['q'][key]
    l['q'][key] += alpha * (y - before)
    l['counts']['planningBackups'] += 1
    return {'pair': key, 'before': before, 'target': y, 'after': l['q'][key]}


def choice(q):
    return 'go' if q['S:go'] > q['S:exit'] else 'exit'


def plan_road(l):
    return [planning_backup(l, 'A:cross'), planning_backup(l, 'S:go')]


def snapshot(l, n):
    rows = distribution(l, 'A:cross')
    return {'n': n, 'successes': sum(o['count'] for o in rows if o['next'] == 'G'),
            'distribution': rows, 'rewardMean': sum(o['probability'] * o['reward'] for o in rows),
            'q': copy.deepcopy(l['q']), 'choice': choice(l['q']), 'counts': dict(l['counts'])}


def acquire(outcomes=('D', 'G', 'G', 'G')):
    l, prefixes = new_learner(), []
    observe(l, EXIT)
    for next_state in outcomes:
        l['counts']['resets'] += 1
        observe(l, GO)
        observe(l, crossing(next_state))
        plan_road(l)
        prefixes.append(dict(snapshot(l, len(prefixes) + 1),
                             realRecords=copy.deepcopy([GO, crossing(next_state)]),
                             acquisition='prescribed-exploration'))
    return l, prefixes


def difference(after, before):
    return {key: after[key] - before[key] for key in COUNTERS}


def actual_episode(l, next_state='D'):
    before, selected = dict(l['counts']), choice(l['q'])
    l['counts']['resets'] += 1
    rows = [GO, crossing(next_state)] if selected == 'go' else [EXIT]
    for row in rows:
        observe(l, row)
    result = sum(GAMMA ** i * row['reward'] for i, row in enumerate(rows))
    plan_road(l)
    return {'firstAction': selected, 'route': [r['s'] for r in rows] + [rows[-1]['next']],
            'rewards': [r['reward'] for r in rows], 'return': result, 'realSteps': len(rows),
            'counts': difference(l['counts'], before),
            'after': snapshot(l, l['model']['A:cross']['n'])}


def frozen_comparisons():
    base, _ = acquire(('D', 'G'))
    branches = {}
    for name, mode, u in (('sampleD', 'sample', F(1, 4)),
                          ('sampleG', 'sample', F(3, 4)),
                          ('expected', 'expected', F(1, 2))):
        l, before = copy.deepcopy(base), dict(base['counts'])
        cross = planning_backup(l, 'A:cross', mode, u)
        go = planning_backup(l, 'S:go')
        q = copy.deepcopy(l['q'])
        counts = difference(l['counts'], before)
        selected = choice(q)
        experience = actual_episode(l)
        branches[name] = {'cross': cross, 'go': go, 'q': q, 'choice': selected,
                          'counts': counts, 'nextExperience': experience}
    return {'frozenQ': base['q'], 'model': base['model'],
            'realCounts': base['counts'], 'branches': branches}


def enumerate_four():
    sequences = []
    # This uses Cartesian product rather than the JS bit-mask enumeration.
    for values in itertools.product(('D', 'G'), repeat=4):
        successes = values.count('G')
        probability = F(3, 4) ** successes * F(1, 4) ** (4 - successes)
        l, _ = acquire(values)
        selected = choice(l['q'])
        sequences.append({'outcomes': list(values), 'successes': successes, 'probability': probability,
                          'choice': selected, 'estimatedGo': l['q']['S:go'],
                          'trueNextEpisodeValue': F(27, 40) if selected == 'go' else F(3, 5),
                          'environmentSteps': l['counts']['environmentSteps']})
    p_go = sum(s['probability'] for s in sequences if s['choice'] == 'go')
    return {'sequences': sequences,
            'successBins': [{'successes': k,
                             'probability': sum(s['probability'] for s in sequences if s['successes'] == k),
                             'choice': 'go' if k >= 3 else 'exit'} for k in range(5)],
            'chooseGoProbability': p_go, 'wrongChoiceProbability': 1 - p_go,
            'expectedNextEpisodeValue': sum(s['probability'] * s['trueNextEpisodeValue'] for s in sequences),
            'expectedNextRealSteps': 1 + p_go}


def walkthrough_data():
    l, prefixes = acquire()
    full = copy.deepcopy(l)
    closure = actual_episode(full)
    stopped, _ = acquire(('D', 'G'))
    cross_before = copy.deepcopy(stopped['model']['A:cross'])
    greedy_episode = actual_episode(stopped)
    before_imagined = dict(stopped['counts'])
    for _ in range(10):
        planning_backup(stopped, 'S:exit', 'sample')
    return {'gamma': GAMMA, 'alpha': 1, 'initialRealRecord': EXIT, 'prefixes': prefixes, 'afterFour': snapshot(l, 4),
            'closure': closure, 'frozen': frozen_comparisons(), 'enumeration': enumerate_four(),
            'greedyTrap': {'crossBefore': cross_before, 'crossAfter': stopped['model']['A:cross'],
                           'greedyEpisode': greedy_episode,
                           'tenImaginedEpisodes': difference(stopped['counts'], before_imagined)}}


def self_test():
    data = walkthrough_data()
    assert [s['rewardMean'] for s in data['prefixes']] == [F(0), F(1, 2), F(2, 3), F(3, 4)]
    assert [s['q']['S:go'] for s in data['prefixes']] == [F(0), F(9, 20), F(3, 5), F(27, 40)]
    assert [s['choice'] for s in data['prefixes']] == ['exit', 'exit', 'exit', 'go']
    assert data['afterFour']['counts'] == dict(zip(COUNTERS, (9, 4, 9, 9, 8, 11, 8)))
    assert data['closure']['after']['rewardMean'] == F(3, 5)
    assert data['closure']['after']['q']['S:go'] == F(27, 50)
    assert data['closure']['after']['choice'] == 'exit'
    trap = data['greedyTrap']
    assert trap['crossBefore'] == trap['crossAfter']
    assert trap['tenImaginedEpisodes'] == dict(zip(COUNTERS, (0, 0, 0, 0, 10, 10, 10)))
    branches = data['frozen']['branches']
    assert [branches[k]['cross']['target'] for k in ('sampleD', 'sampleG', 'expected')] == [F(0), F(1), F(1, 2)]
    assert [branches[k]['counts']['outcomeEvaluations'] for k in ('sampleD', 'sampleG', 'expected')] == [2, 2, 3]
    assert branches['sampleG']['nextExperience']['realSteps'] == 2
    assert branches['expected']['nextExperience']['realSteps'] == 1
    enumeration = data['enumeration']
    assert enumeration['chooseGoProbability'] == F(189, 256)
    assert enumeration['wrongChoiceProbability'] == F(67, 256)
    assert all(s['environmentSteps'] == 9 for s in enumeration['sequences'])
    assert sum(s['probability'] for s in enumeration['sequences']) == 1
    # Independent reward-path arithmetic: this does not call any target/backup.
    paths = ((F(3, 4), (F(0), F(1))), (F(1, 4), (F(0), F(0))))
    true_go = sum(p * sum(GAMMA ** i * r for i, r in enumerate(rewards)) for p, rewards in paths)
    assert true_go == F(27, 40) > F(3, 5)
    expected_consumer = F(189, 256) * true_go + F(67, 256) * F(3, 5)
    assert enumeration['expectedNextEpisodeValue'] == expected_consumer
    # Independent binomial weights, not the learner's recursive model.
    assert [b['probability'] for b in enumeration['successBins']] == [F(n, 256) for n in (1, 12, 54, 108, 81)]
    l, _ = acquire(('D', 'G'))
    before_model = copy.deepcopy(l['model'])
    assert model_target(l, 'A:cross', 'sample', F(1, 2)) == 1  # boundary chooses G
    assert l['model'] == before_model
    q = dict(l['q'])
    assert planning_backup(l, 'A:cross', alpha=F(0))['after'] == q['A:cross']
    assert model_target(l, 'S:go', gamma=F(0)) == 0
    try:
        distribution(new_learner(), 'A:cross')
    except ValueError:
        pass
    else:
        raise AssertionError('Unobserved queries must fail')
    q['G:illegal'] = F(1000)
    assert row_target(q, crossing('G')) == 1
    print('PASS: exact joint counts, budgets, frozen targets, 16 paths, real feedback and boundary checks')


def portable(value):
    if isinstance(value, F):
        return float(value)
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable(item) for item in value]
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    if args.test:
        self_test()
    else:
        print(json.dumps(portable(walkthrough_data()), ensure_ascii=False, indent=2))
