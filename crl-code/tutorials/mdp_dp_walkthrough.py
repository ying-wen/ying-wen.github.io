#!/usr/bin/env python3
"""A known 4x3 model, exact policy prediction and budgeted backups.

Python 3 standard library only. No sampling, training, or imported JS results.
Run --test for analytic/degenerate checks, or --compare path/to/JSON to independently
recompute the published model, policy values, backup snapshots and boundaries.
"""
import argparse
import json
import math

COLS, ROWS = 4, 3
WALL, GOAL = (1, 1), (3, 0)
GAMMA, STEP_REWARD, GOAL_REWARD = .9, -.04, 1.
ACTIONS = ((0, -1), (1, 0), (0, 1), (-1, 0))


def build_model():
    states = [(x, y) for y in range(ROWS) for x in range(COLS)
              if (x, y) not in (WALL, GOAL)]
    model = {}
    for x, y in states:
        outcomes = []
        for dx, dy in ACTIONS:
            next_state = (x + dx, y + dy)
            if not (0 <= next_state[0] < COLS and 0 <= next_state[1] < ROWS) or next_state == WALL:
                next_state = (x, y)
            terminal = next_state == GOAL
            outcomes.append((GOAL_REWARD if terminal else STEP_REWARD,
                             None if terminal else next_state))
        model[(x, y)] = outcomes
    return states, model


def action_values(model, state, values, gamma=GAMMA):
    return [reward + gamma * (0 if next_state is None else values[next_state])
            for reward, next_state in model[state]]


def evaluate_policy(states, model, policy, gamma=GAMMA):
    """Solve (I - gamma P_pi)v = r_pi with pivoted Gaussian elimination."""
    if not 0 <= gamma < 1:
        raise ValueError('requires 0 <= gamma < 1')
    index = {s: i for i, s in enumerate(states)}
    matrix = [[float(i == j) for j in range(len(states))] + [0.] for i in range(len(states))]
    for s, i in index.items():
        for probability, (reward, next_state) in zip(policy[s], model[s]):
            matrix[i][-1] += probability * reward
            if next_state is not None:
                matrix[i][index[next_state]] -= gamma * probability
    n = len(states)
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(matrix[row][col]))
        if abs(matrix[pivot][col]) < 1e-14:
            raise ValueError('singular system')
        matrix[pivot], matrix[col] = matrix[col], matrix[pivot]
        divisor = matrix[col][col]
        matrix[col] = [v / divisor for v in matrix[col]]
        for row in range(n):
            if row != col:
                factor = matrix[row][col]
                matrix[row] = [a - factor * b for a, b in zip(matrix[row], matrix[col])]
    return {s: matrix[i][-1] for s, i in index.items()}


def distances_to_goal(states, model):
    # Reverse breadth-first traversal, separate from numerical value iteration.
    frontier = [None]
    distances = {None: 0}
    while frontier:
        target = frontier.pop(0)
        for state in states:
            if state not in distances and any(next_state == target for _, next_state in model[state]):
                distances[state] = distances[target] + 1
                frontier.append(state)
    return {state: distances[state] for state in states}


def shortest_values(states, distances, gamma=GAMMA):
    return {s: STEP_REWARD * sum(gamma ** i for i in range(distances[s] - 1))
            + gamma ** (distances[s] - 1) * GOAL_REWARD for s in states}


def greedy_policy(states, model, values, gamma=GAMMA):
    policy = {}
    for state in states:
        q = action_values(model, state, values, gamma)
        action = next(i for i, value in enumerate(q) if value >= max(q) - 1e-12)
        policy[state] = [float(i == action) for i in range(4)]
    return policy


def residual(states, model, values, policy=None, gamma=GAMMA):
    errors = []
    for state in states:
        q = action_values(model, state, values, gamma)
        target = max(q) if policy is None else sum(p * v for p, v in zip(policy[state], q))
        errors.append(abs(target - values[state]))
    return max(errors)


def backups(states, model, order, in_place=False, gamma=GAMMA):
    values = {s: 0. for s in states}
    snapshots, rows = [], []
    for sweep in range(6):
        read = values if in_place else values.copy()
        after = values if in_place else values.copy()
        for state in order:
            before = values[state]
            q = action_values(model, state, read, gamma)
            after[state] = max(q)
            rows.append({'backup': len(rows) + 1, 'state': state, 'before': before,
                         'after': after[state], 'action_values': q})
        values = after
        if sweep in (0, 2, 5):
            snapshots.append({'backups': (sweep + 1) * len(states), 'values': values.copy(),
                              'residual': residual(states, model, values, gamma=gamma)})
    return {'order': order, 'in_place': in_place, 'snapshots': snapshots, 'rows': rows}


def discounted_return(rewards, gamma=GAMMA, tail=0.):
    value = tail
    for reward in reversed(rewards):
        value = reward + gamma * value
    return value


def numeric_data():
    states, model = build_model()
    distances = distances_to_goal(states, model)
    uniform = {s: [.25] * 4 for s in states}
    prediction = evaluate_policy(states, model, uniform)
    improved = greedy_policy(states, model, prediction)
    improved_values = evaluate_policy(states, model, improved)
    order = sorted(states, key=lambda s: (distances[s], states.index(s)))
    optimal = shortest_values(states, distances)
    return {'states': states, 'model': model, 'distances': distances, 'optimal': optimal,
            'prediction': prediction, 'improved_policy': improved, 'improved_values': improved_values,
            'synchronous': backups(states, model, states),
            'near_first': backups(states, model, order, True),
            'far_first': backups(states, model, list(reversed(order)), True)}


def near(actual, expected, tolerance=1e-11):
    assert math.isfinite(actual) and abs(actual - expected) <= tolerance, (actual, expected)


def self_test():
    d = numeric_data()
    states, model = d['states'], d['model']
    assert model[(0, 0)][0] == (-.04, (0, 0))  # Boundary self-loop.
    assert model[(1, 0)][2] == (-.04, (1, 0))  # Wall self-loop.
    assert model[(2, 0)][1] == (1., None)  # Arrival reward retained, tail zero.
    near(d['optimal'][(0, 2)], -.04 * (1 + .9 + .81 + .729) + .6561)
    near(d['optimal'][(0, 2)], .51854)
    near(discounted_return([-.04] * 4 + [1]), .51854)
    near(discounted_return([-.04] * 2, tail=d['optimal'][(0, 0)]), .51854)
    near(discounted_return([-.04] * 2), -.076)
    uniform = {s: [.25] * 4 for s in states}
    near(residual(states, model, d['prediction'], uniform), 0)
    near(residual(states, model, d['optimal']), 0)
    near(d['synchronous']['snapshots'][0]['values'][(0, 2)], -.04)
    near(d['near_first']['snapshots'][0]['values'][(0, 2)], .51854)
    near(d['near_first']['snapshots'][0]['residual'], 0)
    assert d['far_first']['snapshots'][0]['residual'] > 0  # The order matters.
    for state in states:
        assert d['improved_values'][state] >= d['prediction'][state] - 1e-12
        near(d['improved_values'][state], d['optimal'][state])
    # gamma=0 discards every successor value; even a wrong table cannot alter targets.
    zero = evaluate_policy(states, model, uniform, gamma=0)
    for state in states:
        near(zero[state], sum(reward for reward, _ in model[state]) / 4)
    near(residual(states, model, d['synchronous']['snapshots'][0]['values'], gamma=0), 0)
    print('PASS: model, hand return, boundaries, policy equation, improvement, equal backups, residual, gamma=0')


def compare_js(path):
    with open(path, encoding='utf8') as handle:
        js = json.load(handle)
    d = numeric_data()
    names = [','.join(map(str, s)) for s in d['states']]
    assert js['states'] == names
    for field, value in [('gamma', GAMMA), ('step_reward', STEP_REWARD), ('goal_reward', GOAL_REWARD)]:
        near(js['config'][field], value)
    for state, name in zip(d['states'], names):
        assert js['distances'][name] == d['distances'][state]
        for a, (reward, next_state) in enumerate(d['model'][state]):
            outcome = js['model'][name][a]
            assert len(outcome) == 1 and outcome[0]['probability'] == 1
            near(outcome[0]['reward'], reward)
            assert outcome[0]['terminal'] == (next_state is None)
            assert outcome[0]['next'] == (None if next_state is None else ','.join(map(str, next_state)))
        for section, field in [('prediction', 'prediction'), ('improvement', 'improved_values')]:
            near(js[section]['values'][name], d[field][state])
        assert js['prediction']['policy'][name] == [.25] * 4
        assert js['improvement']['policy'][name] == d['improved_policy'][state]
        near(js['optimal'][name], d['optimal'][state])
    for method in ('synchronous', 'near_first', 'far_first'):
        actual_method, expected_method = js['propagation'][method], d[method]
        assert actual_method['in_place'] == expected_method['in_place']
        assert actual_method['order'] == [','.join(map(str, s)) for s in expected_method['order']]
        assert len(actual_method['snapshots']) == len(expected_method['snapshots']) == 3
        for expected, actual in zip(expected_method['snapshots'], actual_method['snapshots']):
            assert actual['backups'] == expected['backups']
            near(actual['residual'], expected['residual'])
            for state, name in zip(d['states'], names):
                near(actual['values'][name], expected['values'][state])
        assert len(actual_method['rows']) == len(expected_method['rows']) == 60
        for expected, actual in zip(expected_method['rows'], actual_method['rows']):
            assert actual['backup'] == expected['backup']
            assert actual['state'] == ','.join(map(str, expected['state']))
            near(actual['before'], expected['before'])
            near(actual['after'], expected['after'])
            assert len(actual['action_values']) == len(expected['action_values']) == 4
            for aq, eq in zip(actual['action_values'], expected['action_values']):
                near(aq, eq)
    for field, value in [('full_return', .51854), ('truncated_target', .51854), ('mistaken_terminal', -.076), ('tail', .734)]:
        near(js['boundary'][field], value)
    assert js['budget']['environment_transitions'] == 0 and js['random_rollouts'] == 0
    print('PASS: independently recomputed JS model, values, policies, all snapshots, boundaries')


def json_ready(value):
    if isinstance(value, dict):
        return {','.join(map(str, key)) if isinstance(key, tuple) else key: json_ready(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--compare', metavar='JS_JSON')
    args = parser.parse_args()
    if args.test:
        self_test()
    if args.compare:
        compare_js(args.compare)
    if not args.test and not args.compare:
        print(json.dumps(json_ready(numeric_data()), ensure_ascii=False, indent=2))
