#!/usr/bin/env python3
"""Branching planning with exact Fractions; Python 3 standard library, any cwd.

Run --test for independent path/combinatorial checks, --json for numerical output.
No environment training or random sampling is launched. This original mechanism
is not a reproduction of an author's complete planning algorithm.
"""
import argparse
import itertools
import json
import math
from fractions import Fraction as F

GAMMA = F(9, 10)
LEAVES = ('C1', 'C2', 'C3', 'C4')
REWARDS = (F(2), F(2), F(0), F(0))


def q0(exit_reward=F(3, 5)):
    q = {'S:go': F(0), 'S:exit': exit_reward, 'A:cross': F(0)}
    for s in LEAVES:
        q[s + ':collect'] = q[s + ':wait'] = F(0)
    return q


def choose(q):
    return 'go' if q['S:go'] > q['S:exit'] else 'exit'


def new_counts():
    return dict.fromkeys(('environmentTransitions', 'modelWrites', 'modelQueries',
                         'expandedOutcomes', 'actionValueReads', 'qWrites',
                         'scheduleAppends', 'scheduleBudgetChecks', 'sampleSelections',
                         'queuePushes', 'queuePops', 'work'), 0)


def run(order='backward', mode='expected', draws=(0,), budget=30,
        exit_reward=F(3, 5), g=GAMMA):
    """Direct case implementation; no imports from the JavaScript recurrence."""
    leaf_steps = [(s + ':collect', 'expected', None, F(1)) for s in LEAVES]
    crosses = [('A:cross', 'expected', None, F(1))] if mode == 'expected' else [
        ('A:cross', 'sample', draw, F(1, i + 1)) for i, draw in enumerate(draws)]
    go = ('S:go', 'expected', None, F(1))
    if order == 'forward':
        schedule = [go] + crosses + leaf_steps
    elif order == 'moving':
        assert len(crosses) == 2 and mode == 'sample'
        schedule = [crosses[0]] + leaf_steps + [crosses[1], go]
    else:
        schedule = leaf_steps + crosses + [go]
    q, counts = q0(exit_reward), new_counts()
    counts['scheduleAppends'] = len(schedule)
    snapshots = [dict(q=q.copy(), work=0, leafMean=F(0))]
    stopped = None
    for key, backup_mode, draw, alpha in schedule:
        counts['scheduleBudgetChecks'] += 1
        if key.startswith('C'):
            expanded, reads = 1, 0
        elif key == 'S:go':
            expanded, reads = 1, 1
        elif backup_mode == 'expected':
            expanded, reads = 4, 8
        else:
            assert 0 <= draw < 4
            expanded, reads = 1, 2
        work = 1 + expanded + reads + 1
        if counts['work'] + work > budget:
            stopped = key
            break
        if key.startswith('C'):
            target = REWARDS[LEAVES.index(key.split(':')[0])]
        elif key == 'S:go':
            target = g * q['A:cross']
        elif backup_mode == 'expected':
            target = g * sum(max(q[s + ':collect'], q[s + ':wait']) for s in LEAVES) / 4
        else:
            s = LEAVES[draw]
            target = g * max(q[s + ':collect'], q[s + ':wait'])
        q[key] += alpha * (target - q[key])
        counts['modelQueries'] += 1
        counts['expandedOutcomes'] += expanded
        counts['actionValueReads'] += reads
        counts['qWrites'] += 1
        counts['sampleSelections'] += int(backup_mode == 'sample')
        counts['work'] += work
        snapshots.append(dict(q=q.copy(), work=counts['work'],
                              leafMean=sum(max(q[s + ':collect'], q[s + ':wait']) for s in LEAVES) / 4))
    return dict(q=q, choice=choose(q), counts=counts, snapshots=snapshots,
                unusedBudget=budget-counts['work'], stoppedAt=stopped,
                complete=counts['qWrites'] == len(schedule))


def path_oracle(g=GAMMA):
    """Sum discounted rewards along four complete real paths, no Bellman backup."""
    paths = [(F(0), F(0), r) for r in REWARDS]
    return sum(sum(g ** t * r for t, r in enumerate(path)) for path in paths) / 4


def enumerated(n, budget=None, exit_reward=F(3, 5), order='backward'):
    budget = 16 + 5*n if budget is None else budget
    rows = [run(order, 'sample', draws, budget, exit_reward) for draws in itertools.product(range(4), repeat=n)]
    total = len(rows)
    p_go = F(sum(row['choice'] == 'go' for row in rows), total)
    truth = path_oracle()
    return dict(n=n, enumeratedSequences=total, chooseGoProbability=p_go,
                meanGo=sum(row['q']['S:go'] for row in rows)/total,
                mseGo=sum((row['q']['S:go']-truth)**2 for row in rows)/total,
                meanTrueReturn=p_go*truth+(1-p_go)*exit_reward,
                meanNextRealTransitions=3*p_go+(1-p_go), counts=rows[0]['counts'],
                unusedBudget=rows[0]['unusedBudget'], complete=rows[0]['complete'])


def binomial_oracle(n, exit_reward=F(3, 5)):
    """Independent finite probability law; never runs the value updater."""
    bins = [dict(positive=k, probability=F(math.comb(n, k), 2**n),
                 estimatedGo=2*GAMMA**2*F(k, n)) for k in range(n+1)]
    p_go = sum(b['probability'] for b in bins if b['estimatedGo'] > exit_reward)
    return dict(bins=bins, chooseGoProbability=p_go,
                meanGo=sum(b['probability']*b['estimatedGo'] for b in bins),
                mseGo=sum(b['probability']*(b['estimatedGo']-path_oracle())**2 for b in bins))


def checks():
    backwards, forwards = run(), run('forward')
    assert backwards['q']['S:go'] == path_oracle() == F(81, 100)
    assert forwards['q']['S:go'] == 0 and forwards['choice'] == 'exit'
    assert backwards['counts']['work'] == forwards['counts']['work'] == 30
    assert backwards['counts']['expandedOutcomes'] == 9
    assert backwards['counts']['actionValueReads'] == 9
    assert run(budget=29)['choice'] == 'exit' and run(budget=30)['choice'] == 'go'
    for n in range(1, 5):
        e, b = enumerated(n), binomial_oracle(n)
        for key in ('chooseGoProbability', 'meanGo', 'mseGo'):
            assert e[key] == b[key]
        assert e['meanGo'] == path_oracle()
        assert e['mseGo'] == F(6561, 10000*n)
        assert e['counts']['work'] == 16+5*n
        assert e['counts']['environmentTransitions'] == e['counts']['modelWrites'] == 0
    assert [enumerated(n)['chooseGoProbability'] for n in range(1, 5)] == [F(1, 2), F(3, 4), F(1, 2), F(11, 16)]
    assert enumerated(2, order='moving')['meanGo'] == F(81, 200)
    assert enumerated(2, order='moving')['chooseGoProbability'] == F(1, 2)
    near_tie = enumerated(2, exit_reward=F(41, 50))
    assert near_tie['chooseGoProbability'] == F(1, 4)
    assert near_tie['meanTrueReturn'] == F(327, 400) < F(41, 50)
    assert run(exit_reward=F(81, 100))['choice'] == 'exit'
    assert run(g=F(0))['q']['S:go'] == 0
    assert run(budget=0)['counts']['qWrites'] == 0
    print('PASS: Fraction recurrence, independent reward paths, binomial law, budget and failure cases')


def numeric(x):
    if isinstance(x, F):
        return float(x)
    if isinstance(x, dict):
        return {k: numeric(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [numeric(v) for v in x]
    return x


def data():
    return dict(truth=path_oracle(), order=dict(forward=run('forward'), backward=run()),
                frozen=[enumerated(n) for n in range(1, 5)],
                combinatorial=[binomial_oracle(n) for n in range(1, 5)],
                moving=enumerated(2, order='moving'),
                nearTie=[enumerated(n, exit_reward=F(41, 50)) for n in range(1, 5)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    if args.test:
        checks()
    if args.json:
        print(json.dumps(numeric(data()), ensure_ascii=False, indent=2))
    if not args.test and not args.json:
        checks()
        print('W=30: forward chooses exit; backward chooses go with Q=0.81.')
        print('After frozen leaves: W=21/26/31/36 gives P(go)=1/2,3/4,1/2,11/16.')
