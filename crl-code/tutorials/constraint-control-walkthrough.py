#!/usr/bin/env python3
"""Delivery-route CMDP: exact trajectory and occupancy oracles, stdlib only.

Save this single file in an empty directory. Run it for JSON; --test checks
Fraction trajectory enumeration against an independently solved flow system.
--compare PATH checks generated JavaScript quantities. No random rollouts,
neural training, full CPO implementation, or physical safety system is run.
"""
import argparse
from fractions import Fraction as F
import json
import math

GAMMA = F(9, 10)
BUDGET = F(3, 10)
FAST_REWARD = F(46, 5)


def solve_linear(matrix, rhs):
    """Exact Gauss-Jordan elimination, independent of route occupancy formulas."""
    rows = [list(row) + [value] for row, value in zip(matrix, rhs)]
    for column in range(len(rows)):
        pivot = next(k for k in range(column, len(rows)) if rows[k][column])
        rows[column], rows[pivot] = rows[pivot], rows[column]
        scale = rows[column][column]
        rows[column] = [v / scale for v in rows[column]]
        for k in range(len(rows)):
            if k != column:
                scale = rows[k][column]
                rows[k] = [a - scale * b for a, b in zip(rows[k], rows[column])]
    return [row[-1] for row in rows]


def enumerate_trajectories(p, gamma=GAMMA):
    """Complete two-branch tree with absorbing tails; no sample estimates."""
    if not 0 <= p <= 1 or not 0 <= gamma < 1:
        raise ValueError('invalid policy or discount')
    branches = [(p, [('junction_fast', FAST_REWARD, F(1))]),
                (1 - p, [('junction_detour', F(0), F(0)),
                         ('detour_go', F(10), F(0))])]
    result = {'reward': F(0), 'cost': F(0), 'probability_of_any_cost': F(0)}
    occupancy = dict.fromkeys(['junction_detour', 'junction_fast', 'detour_go', 'terminal_wait'], F(0))
    for mass, path in branches:
        if any(cost > 0 for _, _, cost in path):
            result['probability_of_any_cost'] += mass
        for t, (key, reward, cost) in enumerate(path):
            result['reward'] += mass * gamma ** t * reward
            result['cost'] += mass * gamma ** t * cost
            occupancy[key] += mass * (1 - gamma) * gamma ** t
        occupancy['terminal_wait'] += mass * gamma ** len(path)
    result['occupancy'] = occupancy
    return result


def occupancy_flow(p, gamma=GAMMA):
    # Policy-induced transition rows for junction, detour, terminal.
    transition = [[F(0), 1 - p, p], [F(0), F(0), F(1)], [F(0), F(0), F(1)]]
    matrix = [[F(int(i == j)) - gamma * transition[j][i]
               for j in range(3)] for i in range(3)]
    state_mass = solve_linear(matrix, [1 - gamma, F(0), F(0)])
    return {'junction_detour': state_mass[0] * (1 - p),
            'junction_fast': state_mass[0] * p,
            'detour_go': state_mass[1], 'terminal_wait': state_mass[2]}


def given_trajectory(p, u=F(11, 20)):
    if u < p:
        return [{'state': 'junction', 'action': 'fast', 'reward': FAST_REWARD, 'cost': 1,
                 'next_state': 'terminal', 'terminal': True}]
    return [{'state': 'junction', 'action': 'detour', 'reward': 0, 'cost': 0,
             'next_state': 'detour', 'terminal': False},
            {'state': 'detour', 'action': 'go', 'reward': 10, 'cost': 0,
             'next_state': 'terminal', 'terminal': True}]


def primal_dual_step(p, multiplier, estimated_cost=None, eta_p=F(1, 2), eta_lambda=F(2)):
    cost = p if estimated_cost is None else estimated_cost
    gradient = FAST_REWARD - GAMMA * 10 - multiplier
    raw_p = p + eta_p * gradient
    raw_lambda = multiplier + eta_lambda * (cost - BUDGET)
    return {'old_p': p, 'old_lambda': multiplier, 'estimated_cost': cost,
            'primal_gradient': gradient, 'raw_p': raw_p, 'raw_lambda': raw_lambda,
            'p': min(F(1), max(F(0), raw_p)), 'lambda': max(F(0), raw_lambda)}


def mean_kl(candidate, old):
    return float(1 - GAMMA) * sum(q * math.log(q / r) for q, r in
        [(candidate, old), (1 - candidate, 1 - old)] if q > 0)


def local_step():
    old, budget, delta = .2, .3, .005
    fisher = float(1 - GAMMA) / (old * (1 - old))
    radius = math.sqrt(2 * delta / fisher)
    lower, upper = max(0., old - radius), min(1., old + radius)
    candidate = min(upper, budget)
    return {'old_p': old, 'budget': budget, 'delta': delta, 'fisher': fisher,
            'radius': radius, 'trust_lower': lower, 'trust_upper': upper,
            'reward_only_candidate': upper, 'constrained_candidate': candidate,
            'exact_mean_kl': mean_kl(candidate, old),
            'reward_only_exact_mean_kl': mean_kl(upper, old)}


def metrics(p):
    oracle = enumerate_trajectories(p)
    return {'p': p, 'reward': oracle['reward'], 'cost': oracle['cost'],
            'normalized_reward': (1 - GAMMA) * oracle['reward'],
            'normalized_cost': (1 - GAMMA) * oracle['cost'],
            'probability_of_any_cost': oracle['probability_of_any_cost'],
            'occupancy': occupancy_flow(p)}


def data():
    p, multiplier = F(3, 5), F(0)
    snapshots = [{'k': 0, 'p': p, 'lambda': multiplier, 'trajectory': given_trajectory(p)}]
    updates = []
    for k in range(3):
        step = primal_dual_step(p, multiplier)
        updates.append(step)
        p, multiplier = step['p'], step['lambda']
        snapshots.append({'k': k + 1, 'p': p, 'lambda': multiplier, 'trajectory': given_trajectory(p)})
    return {'kind': 'constructed-exact-cmdp', 'random_rollouts': 0, 'neural_training_steps': 0,
            'exact_update_steps': 3,
            'task': {'gamma': GAMMA, 'initial_distribution': {'junction': 1}, 'budget': BUDGET,
                     'normalized_budget': (1 - GAMMA) * BUDGET, 'fast_reward': FAST_REWARD,
                     'detour_terminal_reward': 10, 'absorbing_reward': 0, 'absorbing_cost': 0},
            'policies': [metrics(p) for p in [F(0), F(3, 10), F(3, 5), F(1)]], 'optimal_p': BUDGET,
            'updates': updates, 'snapshots': snapshots, 'given_action_quantile': F(11, 20),
            'local': local_step(),
            'feasible_curve': [[F(i, 100), enumerate_trajectories(F(i, 100))['reward']] for i in range(101)]}


def numeric(value):
    if isinstance(value, F):
        return float(value)
    if isinstance(value, dict):
        return {k: numeric(v) for k, v in value.items()}
    if isinstance(value, list):
        return [numeric(v) for v in value]
    return value


def compare(a, b, path='root'):
    if isinstance(a, bool) or a is None or isinstance(a, str):
        assert a == b, path
    elif isinstance(a, (float, int)):
        assert math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-11), (path, a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            compare(a[key], b[key], path + '.' + key)
    else:
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, path + '.' + str(i))


def self_test():
    for gamma in [F(0), F(1, 2), GAMMA, F(99, 100)]:
        for p in [F(0), F(1, 5), BUDGET, F(3, 5), F(1)]:
            tree = enumerate_trajectories(p, gamma)
            flow = occupancy_flow(p, gamma)
            assert tree['occupancy'] == flow
            assert sum(flow.values()) == 1
            assert sum(flow[k] * r for k, r in [('junction_fast', FAST_REWARD), ('detour_go', F(10))]) == (1 - gamma) * tree['reward']
            assert flow['junction_fast'] == (1 - gamma) * tree['cost']
    feasible = enumerate_trajectories(BUDGET)
    assert feasible['reward'] == F(453, 50) and feasible['cost'] == BUDGET
    assert feasible['probability_of_any_cost'] == BUDGET
    assert metrics(F(3, 5))['normalized_cost'] == F(3, 50) > F(3, 100)
    first = primal_dual_step(F(3, 5), F(0))
    assert first['p'] == F(7, 10) and first['lambda'] == F(3, 5)
    assert first['estimated_cost'] != first['p']
    assert primal_dual_step(F(1, 10), F(1, 10))['lambda'] == 0
    assert [step['p'] for step in data()['updates']] == [F(7, 10), F(1, 2), F(0)]
    assert given_trajectory(F(3, 5))[0]['action'] == 'fast'
    assert given_trajectory(F(1, 2))[0]['action'] == 'detour'
    local = local_step()
    # Independent grid enumeration of the one-dimensional local optimization.
    eligible = [F(i, 10000) for i in range(10001)
                if .5 * local['fisher'] * (i / 10000 - .2) ** 2 <= .005 + 1e-14 and F(i, 10000) <= BUDGET]
    winner = max(eligible, key=lambda p: enumerate_trajectories(p)['reward'])
    assert winner == BUDGET
    assert local['reward_only_candidate'] > .3
    assert local['reward_only_exact_mean_kl'] < .005
    assert local['exact_mean_kl'] < .005
    print('constraint-control walkthrough independent checks passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--compare', metavar='JSON_PATH')
    args = parser.parse_args()
    if args.test:
        self_test()
    elif args.compare:
        with open(args.compare, encoding='utf-8') as stream:
            compare(numeric(data()), json.load(stream))
        print('Python Fraction oracles and JavaScript quantities agree')
    else:
        print(json.dumps(numeric(data()), ensure_ascii=False, indent=2))
