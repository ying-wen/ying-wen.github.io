#!/usr/bin/env python3
"""Shared tanh quantile predictor: a deterministic, standard-library mechanism.

Save this file in any empty directory. Run it, --test, or --json.
Four frozen-label parent updates and one separate terminal-label successor update.
Known truth is an analyst's controlled diagnostic, not an oracle received by TD.
No RNG, environment rollout, large training job, or third-party dependency.
"""
import argparse
from fractions import Fraction
from itertools import product
import json
import math

TAUS = [(i+.5)/4 for i in range(4)]
FIXED = [1.5]*4
FLOATING = [.5, .5, 2.5, 2.5]


def network(p, x):
    if (not math.isfinite(x) or len(p['w']) != 4 or len(p['b']) != 4
            or not all(math.isfinite(v) for v in [p['a'], p['c']]+p['w']+p['b'])):
        raise ValueError('finite ten-parameter network required')
    h = math.tanh(p['a']*x+p['c'])
    return {'h': h, 'atoms': [b+w*h for b, w in zip(p['b'], p['w'])]}


def fit(fixed, floating):
    return {'a': .5, 'c': .5,
            'w': [(v-f)/math.tanh(1) for f, v in zip(fixed, floating)], 'b': list(fixed)}


def loss(atoms, target):
    # Positive and negative parts instead of the JS signed-residual expression.
    return sum(t*max(y-q, 0)+(1-t)*max(q-y, 0)
               for q, t in zip(atoms, TAUS) for y in target)/(4*len(target))


def gradient(p, x, target):
    pred = network(p, x)
    # Derive head derivatives from strict empirical CDF counts; equality gets zero.
    d = [(sum(y < q for y in target)-t*sum(y != q for y in target))/(4*len(target))
         for q, t in zip(pred['atoms'], TAUS)]
    hidden = sum(v*w for v, w in zip(d, p['w']))*(1-pred['h']**2)
    return dict(pred, loss=loss(pred['atoms'], target), output_gradient=d,
                gradient={'a': x*hidden, 'c': hidden,
                          'w': [v*pred['h'] for v in d], 'b': d})


def step(p, x, target, rate=.5):
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError('positive finite rate required')
    before = gradient(p, x, target)
    g = before['gradient']
    after = {'a': p['a']-rate*g['a'], 'c': p['c']-rate*g['c'],
             'w': [v-rate*d for v, d in zip(p['w'], g['w'])],
             'b': [v-rate*d for v, d in zip(p['b'], g['b'])]}
    return {'rate': rate, 'before': before, 'after': after,
            'after_loss': loss(network(after, x)['atoms'], target)}


def cdf_w1(a, b):
    """Integrate absolute CDF differences in reward space, independently of JS."""
    cuts = sorted(set(a+b))
    return sum((r-l)*abs(sum(v <= l for v in a)/len(a)-sum(v <= l for v in b)/len(b))
               for l, r in zip(cuts, cuts[1:]))


def prediction(p, x, target):
    atoms = network(p, x)['atoms']
    # Variational lower-tail CVaR, unlike JS inverse-CDF integration.
    lower = max(b-sum(max(0, b-v) for v in atoms)/2 for b in atoms)
    return {'atoms': atoms, 'mean': sum(atoms)/4, 'lower_cvar': lower,
            'loss': loss(atoms, target), 'w1': cdf_w1(atoms, target)}


def jacobian(p, x):
    h = network(p, x)['h']
    return [[w*(1-h*h)*x, w*(1-h*h)]
            +[h if i == j else 0 for j in range(4)]
            +[1 if i == j else 0 for j in range(4)] for i, w in enumerate(p['w'])]


def samples():
    # Enumerate the four binary outcomes directly. Reverse bit order matches JSON order.
    sequences = []
    for outcomes in product([0, 4], repeat=4):
        rewards = list(reversed(outcomes))
        k = rewards.count(0)
        sequences.append({'rewards': rewards, 'returns': [.5+.5*r for r in rewards],
                          'count_low': k, 'empirical_cdf_at_one': k/4,
                          'absolute_cdf_error': abs(k/4-.5), 'mass': 1/16})
    return {'specified': [0, 0, 0, 4], 'specified_returns': [.5, .5, .5, 2.5],
            'specified_cdf_at_one': .75, 'true_cdf_at_one': .5, 'sequences': sequences,
            'counts': [sum(s['count_low'] == k for s in sequences) for k in range(5)],
            'expected_cdf': sum(s['empirical_cdf_at_one']/16 for s in sequences),
            'expected_absolute_cdf_error': sum(s['absolute_cdf_error']/16 for s in sequences),
            'expected_squared_cdf_error': sum((s['empirical_cdf_at_one']-.5)**2/16 for s in sequences)}


def data():
    initial = fit(FIXED, [1, 1, 2, 2])
    p, snapshots = initial, []
    for k in range(5):
        snapshots.append({'updates': k, 'parameters': p,
                          'fixed': prediction(p, -1, FIXED),
                          'floating': prediction(p, 1, FLOATING)})
        if k < 4:
            p = step(p, 1, FLOATING)['after']
    successor = fit([2]*4, [1, 1, 3, 3])
    moved = step(successor, 1, [0, 0, 4, 4])
    v0, v1 = network(successor, 1)['atoms'], network(moved['after'], 1)['atoms']
    y0, y1 = [[.5+.5*z for z in v] for v in [v0, v1]]
    left, right = jacobian(initial, -1), jacobian(initial, 1)
    return {'kind': 'exact-shared-tanh-quantile-mechanism', 'random_rollouts': 0,
            'optimizer_updates': 4, 'successor_optimizer_updates': 1, 'stochastic_training_runs': 0,
            'task': {'gamma': .5, 'advance': .5, 'fixed_final_reward': 2,
                     'floating_final_rewards': [0, 4], 'floating_final_masses': [.5, .5]},
            'inputs': {'fixed': -1, 'floating': 1}, 'taus': TAUS,
            'true_fixed': FIXED, 'true_floating': FLOATING, 'initial': initial,
            'first_step': step(initial, 1, FLOATING), 'snapshots': snapshots,
            'exact_fit': fit(FIXED, FLOATING),
            'initial_cross_jacobian': [[sum(a*b for a, b in zip(l, r)) for r in right] for l in left],
            'aliased_inputs': {'input': 1, 'truth_distance': 1, 'minimum_sum_w1_lower_bound': 1},
            'bootstrap': {'successor_before': v0, 'successor_after': v1,
                          'target_before': y0, 'target_after': y1, 'frozen_target': y0,
                          'held_parent': [1, 1, 2, 2], 'parent_loss_before': loss([1, 1, 2, 2], y0),
                          'parent_loss_after': loss([1, 1, 2, 2], y1),
                          'target_w1_before': cdf_w1(y0, FLOATING), 'target_w1_after': cdf_w1(y1, FLOATING),
                          'successor_step': moved}, 'finite_sample': samples()}


def tests():
    checks = 0

    def near(a, b, tolerance=1e-9):
        nonlocal checks
        assert abs(a-b) <= tolerance, (a, b)
        checks += 1

    d = data()
    # Enumerate finite task paths without using the network or Bellman-target code.
    paths = [('fixed', [.5, 2], Fraction(1)),
             ('floating', [.5, 0], Fraction(1, 2)),
             ('floating', [.5, 4], Fraction(1, 2))]
    for action in ['fixed', 'floating']:
        mean = sum(m*sum(Fraction(str(r))*Fraction(1, 2)**t for t, r in enumerate(rs))
                   for a, rs, m in paths if a == action)
        near(float(mean), 1.5)
    near(cdf_w1(FIXED, FLOATING), 1)
    exact = d['exact_fit']
    for x, truth in [(-1, FIXED), (1, FLOATING)]:
        for v, z in zip(network(exact, x)['atoms'], truth):
            near(v, z)
    # Two loss evaluations per parameter give an independent derivative check.
    for p in [d['initial'], d['snapshots'][2]['parameters']]:
        g = gradient(p, 1, FLOATING)['gradient']
        for key, i in [('a', None), ('c', None)]+[(k, i) for k in ['w', 'b'] for i in range(4)]:
            plus = {k: list(v) if isinstance(v, list) else v for k, v in p.items()}
            minus = {k: list(v) if isinstance(v, list) else v for k, v in p.items()}
            h = 1e-6
            if i is None:
                plus[key] += h
                minus[key] -= h
                expected = g[key]
            else:
                plus[key][i] += h
                minus[key][i] -= h
                expected = g[key][i]
            near((loss(network(plus, 1)['atoms'], FLOATING)-loss(network(minus, 1)['atoms'], FLOATING))/(2*h), expected)
    for i, snap in enumerate(d['snapshots']):
        near(snap['fixed']['mean'], 1.5)
        near(snap['floating']['mean'], 1.5)
        near(snap['fixed']['w1'], i/32)
        near(snap['parameters']['a'], snap['parameters']['c'])
        if i:
            assert snap['floating']['loss'] < d['snapshots'][i-1]['floating']['loss']
            assert snap['floating']['w1'] < d['snapshots'][i-1]['floating']['w1']
    # All 16 data sets: CDF variance, unlike a fitted-network calibration claim.
    s = d['finite_sample']
    assert s['counts'] == [1, 4, 6, 4, 1]
    near(s['expected_cdf'], .5)
    near(s['expected_absolute_cdf_error'], float(Fraction(3, 16)))
    near(s['expected_squared_cdf_error'], float(Fraction(1, 16)))
    # Expected sample gradient equals the population gradient (away from zero residuals).
    whole = gradient(d['initial'], 1, FLOATING)['output_gradient']
    for i in range(4):
        near(sum(gradient(d['initial'], 1, [y])['output_gradient'][i] for y in [.5, 2.5])/2, whole[i])
    b = d['bootstrap']
    assert b['parent_loss_after'] > b['parent_loss_before']
    assert b['target_w1_after'] < b['target_w1_before']
    near(cdf_w1(b['target_before'], b['target_after']), .059511235292667414)
    print(f'{checks} independent neural-distribution checks passed; no RNG.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    if args.test:
        tests()
    elif args.json:
        print(json.dumps(data(), ensure_ascii=False))
    else:
        d = data()
        for s in d['snapshots']:
            print(f"k={s['updates']}: fixed={s['fixed']['atoms']}; floating={s['floating']['atoms']}")
            print(f"  W1 fixed={s['fixed']['w1']:.6f}, floating={s['floating']['w1']:.6f}; both means=1.5")
        print('A successor write moves the parent target:', d['bootstrap']['target_after'])
        print('Four outcomes: expected |empirical CDF(1)-.5| = 3/16.')
        print('No RNG; deterministic controlled mechanism, not a stochastic training benchmark.')


if __name__ == '__main__':
    main()
