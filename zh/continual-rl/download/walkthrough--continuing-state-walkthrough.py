#!/usr/bin/env python3
"""Continuing door loop: exact finite-window checks; Python 3.10+ standard library.

This independent reference expands input-to-state paths with Fraction. It does
not import the website, simulate a training sweep, or expose hidden z to a learner.
The z-dependent rewards below belong to the explicitly known analysis model.
"""
import argparse
from fractions import Fraction as F
from itertools import product
import json
import math
import unittest


def expand(inputs, versions, offset=0):
    """Closed sum of input paths, not h/S forward recurrence."""
    return sum(x * math.prod(a + offset for a in versions[i + 1:])
               for i, x in enumerate(inputs))


def derivative(inputs, versions):
    return sum(x * math.prod(versions[k] for k in range(i + 1, len(inputs)) if k != j)
               for i, x in enumerate(inputs) for j in range(i + 1, len(inputs)))


def analyze(z, cues, *, a=F(-1, 2), alpha=F(1, 2), erase_cue=False,
            boundary='carry', fixed_action=None):
    # Model analysis: independently expand each decision, then write its parameter.
    inputs, versions, rows, activity = [], [], [], []
    h = s = F(0)

    def observe(x, place, step):
        nonlocal h, s
        inputs.append(F(x)); versions.append(a)
        h, s = expand(inputs, versions), derivative(inputs, versions)
        activity.append(dict(step=step, place=place, input=F(x), h=h,
                             sensitivity=s, usedParameter=a))

    for k, cue in enumerate(cues):
        observe(0 if erase_cue else cue, 'C', 3*k)
        observe(0, 'J', 3*k+1)
        d = fixed_action if fixed_action is not None else (1 if h >= 0 else -1)
        q, dq, reward = d*h, d*s, d*z
        row = dict(decisionState=h, decisionSensitivity=s, action=d, prediction=q,
                   derivative=dq, usedParameter=a, reward=reward)
        observe(reward, 'L' if d == 1 else 'R', 3*k+2)
        g = (q-reward)*dq
        a -= alpha*g
        row.update(gradient=g, arrivedState=h, arrivedSensitivity=s, updatedParameter=a)
        rows.append(row)
        if k == 0 and boundary != 'carry':
            # New independent constant boundary; derivative of its numeric h is zero.
            inputs, versions = [h if boundary == 'detach' else F(0)], [F(0)]
    return dict(a=a, h=h, sensitivity=s, decisions=rows, activity=activity,
                totalReward=sum(r['reward'] for r in rows), environmentSteps=3*len(cues)-1,
                initializationCalls=1, resetCalls=0, terminated=False)


def enumerate_paths(p=F(4, 5), **options):
    paths = []
    for z, c1, c2 in product([-1, 1], repeat=3):
        mass = F(1, 2)*(p if c1 == z else 1-p)*(p if c2 == z else 1-p)
        paths.append(dict(z=z, cues=[c1,c2], probability=mass, **analyze(z,[c1,c2],**options)))
    rewards = [sum(r['probability']*r['decisions'][i]['reward'] for r in paths) for i in [0,1]]
    return dict(paths=paths, expectedRewards=rewards, expectedTotal=sum(rewards),
                secondSuccess=sum(r['probability'] for r in paths if r['decisions'][1]['reward']==1))


def conditional_expectation(p=F(4,5), a=F(-1,2), alpha=F(1,2)):
    # A second independent derivation: conditional tree over the first outcome.
    result = [F(0), F(0)]
    for z in [-1,1]:
        for c in [-1,1]:
            prob = F(1,2)*(p if c==z else 1-p)
            d = 1 if a*c >= 0 else -1
            r = d*z
            a1 = a-alpha*(d*a*c-r)*d*c
            result[0] += prob*r
            for nxt in [-1,1]:
                h2 = a1*a1*(a*a*c+r)+a1*nxt
                result[1] += prob*(p if nxt==z else 1-p)*(1 if h2>=0 else -1)*z
    return result


def jsonable(value):
    if isinstance(value, F): return float(value)
    if isinstance(value, dict): return {k:jsonable(v) for k,v in value.items()}
    if isinstance(value, list): return [jsonable(v) for v in value]
    return value


def data():
    return dict(live=analyze(1,[1,1]), detached=analyze(1,[1,1],boundary='detach'),
                erased=analyze(1,[1,1],boundary='erase'), enumeration=enumerate_paths(),
                frozen=enumerate_paths(alpha=F(0)), noCue=enumerate_paths(erase_cue=True),
                largeStep=enumerate_paths(alpha=F(2)), noInformation=enumerate_paths(p=F(1,2)))


class Checks(unittest.TestCase):
    def test_exact_history(self):
        r=analyze(1,[1,1]); d=r['decisions'][1]
        self.assertEqual([x['h'] for x in r['activity']],list(map(F,[1,F(-1,2),F(-3,4),F(13,16),F(13,64),F(269,256)])))
        self.assertEqual(d['decisionSensitivity'],F(9,16))
        self.assertEqual(d['gradient'],F(-459,1024))
        self.assertEqual(r['a'],F(971,2048))

    def test_finite_difference_fixed_schedule(self):
        for path in enumerate_paths()['paths']:
            rows=path['activity'][:5]
            inputs=[r['input'] for r in rows]; versions=[r['usedParameter'] for r in rows]
            eps=F(1,10**6)
            state_fd=(expand(inputs,versions,eps)-expand(inputs,versions,-eps))/(2*eps)
            d=path['decisions'][1]
            def loss(e): return (d['action']*expand(inputs,versions,e)-d['reward'])**2/2
            loss_fd=(loss(eps)-loss(-eps))/(2*eps)
            self.assertAlmostEqual(float(state_fd),float(d['decisionSensitivity']),places=8)
            self.assertAlmostEqual(float(loss_fd),float(d['gradient']),places=8)

    def test_conditional_tree(self):
        for p in [F(0),F(1,2),F(4,5),F(1)]:
            for alpha in [F(0),F(1,2),F(2)]:
                self.assertEqual(enumerate_paths(p,alpha=alpha)['expectedRewards'],conditional_expectation(p,alpha=alpha))
        self.assertEqual(enumerate_paths()['expectedRewards'],[F(-3,5),F(9,25)])

    def test_reward_information(self):
        r=enumerate_paths(erase_cue=True)
        self.assertEqual(r['expectedRewards'],[0,1])
        # No first gradient, yet a retained action/reward history can inform control.
        self.assertTrue(all(x['decisions'][0]['gradient']==0 for x in r['paths']))

    def test_boundary_and_failure(self):
        full=analyze(1,[1,1]); detached=analyze(1,[1,1],boundary='detach'); erased=analyze(1,[1,1],boundary='erase')
        self.assertEqual(full['decisions'][1]['decisionState'],detached['decisions'][1]['decisionState'])
        self.assertEqual(detached['decisions'][1]['decisionSensitivity'],F(5,8))
        self.assertEqual(erased['decisions'][1]['decisionState'],F(1,4))
        self.assertEqual(enumerate_paths(alpha=F(2))['expectedRewards'][1],0)
        self.assertGreater(analyze(-1,[1,-1])['a'],1)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test',action='store_true');parser.add_argument('--json',action='store_true')
    args=parser.parse_args()
    if args.test: unittest.main(argv=['continuing-state-walkthrough.py'])
    elif args.json: print(json.dumps(jsonable(data()),indent=2))
    else:
        r=analyze(1,[1,1])
        print('One initialization; 5 real transitions; 0 resets; no terminal event.')
        for row in r['activity']:
            print('t={step} {place}: a={usedParameter}, h={h}, S={sensitivity}'.format(**row))
        for i,row in enumerate(r['decisions'],1):
            print(f"door {i}: action={row['action']}, reward={row['reward']}, gradient={row['gradient']}, a+={row['updatedParameter']}")
        e=enumerate_paths()
        print('Exact 8-path rewards:',e['expectedRewards'],'sum:',e['expectedTotal'])
        print('No cues, reward still informative:',enumerate_paths(erase_cue=True)['expectedRewards'])
        print('alpha=2 second reward:',enumerate_paths(alpha=F(2))['expectedRewards'][1])
        print('This is an observation window; the continuing process has no terminal boundary here.')
