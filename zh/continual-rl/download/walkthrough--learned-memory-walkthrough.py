#!/usr/bin/env python3
"""Learn one write gate. Python 3.10+, standard library; no random training.

Run from any directory: python3 learned-memory-walkthrough.py [--test | --json]
Every visit uses a fixed eta. Carried activity is a constant boundary for that
visit's derivative. Updates do not reconstruct activity or differentiate actions.
"""
import argparse
import itertools
import json
import math

DELAYS = (4, 8, 4, 4, 8, 4)
CUES = (1, 1, -1, 1, 1, -1)
ETA = math.log(1 / 3)


def sigmoid(x):
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def path(draws, alpha=1, eta=ETA, skip=-1):
    """Independent passage calculation; delays affect real costs, not h (rho=1)."""
    h, steps, rows = 0.0, -1, []
    for i, (z, cue, delay) in enumerate(draws):
        old, gate = h, sigmoid(eta)
        h = old + gate * (cue - old)
        action = 1 if h >= 0 else -1
        reward = action * z
        sensitivity = (cue - old) * gate * (1 - gate)
        gradient = (h - action * reward) * sensitivity
        used_eta = eta
        if i != skip:
            eta -= alpha * gradient
        steps += delay + 2
        rows.append(dict(visit=i+1, oldH=old, cue=cue, gate=gate, h=h, S=sensitivity,
                         action=action, reward=reward, gradient=gradient, usedEta=used_eta,
                         updatedEta=eta, nextGate=sigmoid(eta), rewardStep=steps))
    return dict(rows=rows, totalReward=sum(r['reward'] for r in rows), environmentSteps=steps)


def enumerate_paths(accuracy=.8, **kwargs):
    totals, gates, losses = [0.0]*6, [0.0]*6, [0.0]*6
    probability = 0.0
    outcomes = tuple(itertools.product((-1, 1), repeat=2))
    for choices in itertools.product(outcomes, repeat=6):
        mass = math.prod(.5*(accuracy if z == c else 1-accuracy) for z, c in choices)
        draws = [(z, c, d) for (z, c), d in zip(choices, DELAYS)]
        result = path(draws, **kwargs)
        probability += mass
        for i, row in enumerate(result['rows']):
            totals[i] += mass*row['reward']
            gates[i] += mass*row['gate']
            losses[i] += mass*.5*(row['action']*row['h']-row['reward'])**2
    return dict(pathCount=4096, probability=probability, expectedRewards=totals,
                expectedGate=gates, expectedLoss=losses, expectedTotal=sum(totals))


def loss(eta, old, cue, action, reward):
    g = sigmoid(eta)
    return .5*(action*((1-g)*old+g*cue)-reward)**2


def tests():
    draws = list(zip(CUES, CUES, DELAYS))
    live, omitted = path(draws), path(draws, skip=2)
    assert live['environmentSteps'] == 43
    assert [r['rewardStep'] for r in live['rows']] == [5,15,21,27,37,43]
    assert live['rows'][2]['reward'] == -1
    assert (live['rows'][-1]['action'], omitted['rows'][-1]['action']) == (-1, 1)
    assert (live['totalReward'], omitted['totalReward']) == (4, 2)
    for row in live['rows']:
        eps = 1e-5
        args = (row['oldH'], row['cue'], row['action'], row['reward'])
        fd = (loss(row['usedEta']+eps,*args)-loss(row['usedEta']-eps,*args))/(2*eps)
        assert math.isclose(fd, row['gradient'], abs_tol=2e-10)
    # Independently expand the whole realized cue history with recorded gates.
    for i, row in enumerate(live['rows']):
        expanded = sum(r['cue']*r['gate']*math.prod(1-s['gate'] for s in live['rows'][j+1:i+1])
                       for j,r in enumerate(live['rows'][:i+1]))
        assert math.isclose(expanded,row['h'],abs_tol=1e-14)
    # Conditional expected square loss = variance/2 + squared bias/2.
    for old, cue, eta in itertools.product((-.8,0,.7),(-1,1),(-8,ETA,0,8)):
        h = (1-sigmoid(eta))*old+sigmoid(eta)*cue
        exact = .8*.5*(h-cue)**2+.2*.5*(h+cue)**2
        assert math.isclose(exact,.32+.5*(h-.6*cue)**2,abs_tol=1e-14)
    assert abs(enumerate_paths(accuracy=.5)['expectedTotal']) < 1e-12
    assert math.isclose(enumerate_paths(alpha=0)['expectedTotal'],2.7,abs_tol=1e-10)
    assert abs(path(draws,eta=-8)['rows'][0]['gradient']) < .001
    return 'PASS: finite differences, expanded history, conditional loss, 4096 paths, saturation, action intervention'


def data():
    draws = list(zip(CUES,CUES,DELAYS))
    return dict(live=path(draws),omitted=path(draws,skip=2),frozen=path(draws,alpha=0),
                expectation=enumerate_paths(),omittedExpectation=enumerate_paths(skip=2),
                frozenExpectation=enumerate_paths(alpha=0),uninformativeExpectation=enumerate_paths(accuracy=.5))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test',action='store_true')
    parser.add_argument('--json',action='store_true')
    args = parser.parse_args()
    if args.test:
        print(tests())
    elif args.json:
        print(json.dumps(data(),indent=2))
    else:
        result = data()
        print('六次真实到达：所有门奖励都学习；窗口43步，末门后尚可继续返回。')
        print('visit  cue   old-h      gate       h          action reward gradient')
        for r in result['live']['rows']:
            print(f"{r['visit']:5d} {r['cue']:+4d} {r['oldH']:+.6f} {r['gate']:.6f} {r['h']:+.6f} {r['action']:+7d} {r['reward']:+6d} {r['gradient']:+.6f}")
        for name in ('expectation','omittedExpectation','frozenExpectation','uninformativeExpectation'):
            print(name,result[name]['expectedRewards'],result[name]['expectedTotal'])
        print('数值为4096条有限路径的加权枚举；不包含随机或长时程训练。')
