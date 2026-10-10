#!/usr/bin/env python3
"""Fixed support hole, CQL gradients, IQL and OPE; Python standard library only.

Run normally for JSON, --test for independent gradients/limits/nonidentifiability,
or --compare public/crl-figures/offline-support-data.json for JS/Python parity.
The four trajectories and Q snapshot are given inputs. No random sampling,
environment interaction, network fitting, or benchmark training is performed.
"""
import argparse
import json
import math

GAMMA = .9
BEHAVIOR = [.5, .5, 0.]
GIVEN_Q = [0., 2., 4.]
TRUE_REWARDS = [0., 2., -2.]


def value_target(reward, terminal, next_value):
    return reward if terminal else reward + GAMMA * next_value


def check_probability(probability):
    if (not probability or any(not math.isfinite(p) or p < 0 for p in probability)
            or abs(sum(probability) - 1) > 1e-12):
        raise ValueError('probabilities must sum to one')


def check_support(policy, behavior=BEHAVIOR):
    check_probability(policy)
    check_probability(behavior)
    if len(policy) != len(behavior) or any(p > 0 and m == 0 for p, m in zip(policy, behavior)):
        raise ValueError('target has an unsupported action')


def softmax(values):
    peak = max(values)
    weights = [math.exp(v - peak) for v in values]
    return [w / sum(weights) for w in weights]


def expectile(values, probability, tau):
    """Bisection of the first-order condition, ignoring zero-mass actions."""
    check_probability(probability)
    if (len(values) != len(probability) or not 0 < tau < 1
            or any(p > 0 and not math.isfinite(x) for x, p in zip(values, probability))):
        raise ValueError('finite supported values and 0 < tau < 1 required')
    supported = [q for q, p in zip(values, probability) if p > 0]
    low, high = min(supported), max(supported)
    for _ in range(90):
        value = (low + high) / 2
        balance = sum(p * (1 - tau if q < value else tau) * (q - value)
                      for q, p in zip(values, probability) if p > 0)
        if balance > 0:
            low = value
        else:
            high = value
    return (low + high) / 2


def expectile_loss(value, values=GIVEN_Q, probability=BEHAVIOR, tau=.75):
    return sum(p * (1 - tau if q < value else tau) * (q - value) ** 2
               for q, p in zip(values, probability) if p > 0)


def extracted_policy(values=GIVEN_Q, probability=BEHAVIOR, value=1.5, beta=1.):
    """Exact weighted categorical likelihood optimum, allowing simplex boundary."""
    check_probability(probability)
    if len(values) != len(probability) or not math.isfinite(beta) or beta < 0:
        raise ValueError('nonnegative finite inverse temperature required')
    peak = max(beta * (q - value) for q, p in zip(values, probability) if p > 0)
    weights = [0. if p == 0 else p * math.exp(beta * (q - value) - peak) for q, p in zip(values, probability)]
    return [w / sum(weights) for w in weights]


def cql_loss(values, alpha=1., state_mass=.5):
    """s contribution to global mean; terminal Bellman targets are frozen."""
    peak = max(values)
    logsum = peak + math.log(sum(math.exp(q - peak) for q in values))
    bellman = .5 * sum(p * (q - r) ** 2 for p, q, r in zip(BEHAVIOR, values, TRUE_REWARDS) if p > 0)
    regularizer = logsum - sum(p * q for p, q in zip(BEHAVIOR, values))
    return state_mass * (bellman + alpha * regularizer)


def cql_gradient(values, alpha=1., state_mass=.5):
    return [state_mass * ((0. if p == 0 else p * (q - r)) + alpha * (s - p))
            for p, q, r, s in zip(BEHAVIOR, values, TRUE_REWARDS, softmax(values))]


def dataset(unseen_reward=-2.):
    return [[{'state': 'start', 'action': 'go', 'reward': 0., 'next_state': 's', 'terminal': False, 'mu': 1.},
             {'state': 's', 'action': action, 'reward': [0., 2., unseen_reward][action],
              'next_state': None, 'terminal': True, 'mu': .5}]
            for action in (0, 0, 1, 1)]


def pdis(episodes, policy):
    check_support(policy)  # Checking only actions appearing in a log misses the hole.
    results = []
    for episode in episodes:
        weight, value = 1., 0.
        for t, step in enumerate(episode):
            weight *= 1. if step['state'] == 'start' else policy[step['action']] / step['mu']
            value += GAMMA ** t * weight * step['reward']
        results.append(value)
    return sum(results) / len(results)


def doubly_robust(episodes, policy, values=GIVEN_Q):
    check_support(policy)
    value = sum(p * q for p, q in zip(policy, values))
    q_start = GAMMA * value
    results = []
    for episode in episodes:
        result = 0.
        for step in reversed(episode):
            qs = q_start if step['state'] == 'start' else values[step['action']]
            vs = q_start if step['state'] == 'start' else value
            ratio = 1. if step['state'] == 'start' else policy[step['action']] / step['mu']
            result = vs + ratio * (step['reward'] + GAMMA * result - qs)
        results.append(result)
    return sum(results) / len(results)


def numeric_data():
    values = GIVEN_Q.copy()
    probability = softmax(values)
    gradient = cql_gradient(values)
    next_values = [q - g for q, g in zip(values, gradient)]
    value = expectile(values, BEHAVIOR, .75)
    policy = extracted_policy(values, BEHAVIOR, value)
    episodes = dataset()
    candidate = [.1, .8, .1]
    return {
        'kind': 'constructed-exact-example', 'random_rollouts': 0, 'optimizer_steps': 0, 'hypothetical_gradient_steps': 1,
        'task': {'gamma': GAMMA, 'states': ['start', 's'], 'start_action': 'go', 'start_reward': 0.,
                 'terminal_rewards': TRUE_REWARDS, 'unseen_reward_known_to_learner': False},
        'dataset': {'episodes': episodes, 'episode_count': 4, 'transition_count': 8,
                    'state_mass': {'start': .5, 's': .5}, 'action_counts': [2, 2, 0], 'behavior': BEHAVIOR},
        'bootstrap': {'q': values, 'greedy_action': 2, 'observed_terminal_squared_error': 0.,
                      'start_target': value_target(0., False, max(values)), 'true_greedy_return': GAMMA * TRUE_REWARDS[2],
                      'hypothetical_online_feedback': {'action': 2, 'reward': -2., 'terminal': True,
                                                       'step_size': 1., 'next_q': [0., 2., -2.], 'next_start_target': 1.8}},
        'cql': {'alpha': 1., 'state_mass': .5, 'step_size': 1., 'softmax': probability,
                'conditional_regularizer_gradient': [s - p for s, p in zip(probability, BEHAVIOR)],
                'full_mean_gradient': gradient, 'next_q': next_values, 'next_start_target': GAMMA * max(next_values),
                'loss_before': cql_loss(values), 'loss_after': cql_loss(next_values)},
        'iql': {'tau': .75, 'beta': 1., 'value': value, 'value_loss': expectile_loss(value),
                'q_start_target': value_target(0., False, value), 'advantages': [None if p == 0 else q - value for q, p in zip(values, BEHAVIOR)],
                'actor_probability': policy, 'true_actor_return': GAMMA * sum(p * r for p, r in zip(policy, TRUE_REWARDS)),
                'expectile_curve': [[-.5 + i * 3 / 80, expectile_loss(-.5 + i * 3 / 80)] for i in range(81)],
                'tau_examples': [[tau, expectile(values, BEHAVIOR, tau)] for tau in (.5, .75, .9, .99, .999999)]},
        'ope': {'fixed_policy': policy, 'pdis': pdis(episodes, policy), 'dr_exact_supported_model': doubly_robust(episodes, policy),
                'dr_zero_model': doubly_robust(episodes, policy, [0., 0., 0.]), 'unsupported_candidate': candidate,
                'same_logs_in_both_worlds': dataset(-2.) == dataset(4.),
                'true_candidate_returns': [GAMMA * (candidate[1] * 2 + candidate[2] * r) for r in (-2., 4.)],
                'naive_missing_mass_pdis': GAMMA * candidate[1] * 2,
                'naive_uncorrected_model': GAMMA * (candidate[1] * 2 + candidate[2] * 4)},
    }


def near(actual, expected, tolerance=1e-10):
    assert math.isfinite(actual) and abs(actual - expected) <= tolerance, (actual, expected)


def self_test():
    data = numeric_data()
    near(data['bootstrap']['start_target'], 3.6)
    near(data['bootstrap']['true_greedy_return'], -1.8)
    assert all(episode[0]['terminal'] is False and episode[1]['terminal'] is True for episode in dataset())
    near(value_target(2., True, math.nan), 2.)  # Real terminal: do not read the poisoned tail.
    near(value_target(2., False, 1.5), 3.35)  # A truncation with a continuing task retains a tail.
    assert len(dataset()) * 2 == 8 and dataset(-2.) == dataset(4.)
    # Full loss derivatives at both zero and nonzero Bellman residuals.
    for values in (GIVEN_Q, [.7, 1.1, 3.2]):
        for alpha, mass in ((1., .5), (0., .5), (.3, 1.)):
            for i, derivative in enumerate(cql_gradient(values, alpha, mass)):
                plus, minus = values.copy(), values.copy()
                plus[i] += 1e-5
                minus[i] -= 1e-5
                near((cql_loss(plus, alpha, mass) - cql_loss(minus, alpha, mass)) / 2e-5, derivative, 5e-9)
    near(sum(data['cql']['full_mean_gradient']), 0.)
    assert data['cql']['next_q'][0] > 0 and data['cql']['next_q'][2] > -2
    assert data['cql']['next_q'][2] < 4 and data['cql']['loss_after'] < data['cql']['loss_before']
    # Independent closed form: equal masses at 0 and 2 yield v=2*tau.
    for tau in (.1, .5, .75, .9, .99, .999999):
        near(expectile(GIVEN_Q, BEHAVIOR, tau), 2 * tau)
    near(expectile([0., 2., 1000.], BEHAVIOR, .75), 1.5)
    near(expectile([0., 2., math.nan], BEHAVIOR, .75), 1.5)
    near(expectile([3., 3.], [.2, .8], .75), 3.)
    near((expectile_loss(1.5 + 1e-5) - expectile_loss(1.5 - 1e-5)) / 2e-5, 0.)
    near(extracted_policy(beta=0.)[1], .5)
    near(extracted_policy(beta=50.)[1], 1.)
    assert extracted_policy()[2] == 0
    assert extracted_policy([0., 2., 1000.])[2] == 0
    assert extracted_policy([0., 2., math.nan])[2] == 0
    near(extracted_policy(value=-10.)[1], extracted_policy(value=10.)[1])  # State baseline cancels at exact categorical optimum.
    near(data['iql']['q_start_target'], 1.35)
    near(data['iql']['actor_probability'][1], math.exp(2) / (1 + math.exp(2)))
    for key in ('pdis', 'dr_exact_supported_model', 'dr_zero_model'):
        near(data['ope'][key], data['iql']['true_actor_return'])
    near(data['ope']['true_candidate_returns'][0], 1.26)
    near(data['ope']['true_candidate_returns'][1], 1.8)
    for estimator in (pdis, doubly_robust):
        try:
            estimator(dataset(), [.1, .8, .1])
        except ValueError:
            pass
        else:
            raise AssertionError('support must be checked before using logged ratios')
    for tau in (0., 1.):
        try:
            expectile(GIVEN_Q, BEHAVIOR, tau)
        except ValueError:
            pass
        else:
            raise AssertionError('expectile endpoints must not silently change the minimizer')
    return 'offline support: gradients, limits, terminal fields, OPE and compatible-world checks passed'


def compare(actual, expected, path='root'):
    if isinstance(actual, bool) or actual is None or isinstance(actual, str):
        assert actual == expected, path
    elif isinstance(actual, (float, int)):
        near(actual, expected)
    elif isinstance(actual, list):
        assert len(actual) == len(expected), path
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, f'{path}[{i}]')
    else:
        assert set(actual) == set(expected), path
        for key in actual:
            compare(actual[key], expected[key], f'{path}.{key}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--compare')
    arguments = parser.parse_args()
    if arguments.test:
        print(self_test())
    elif arguments.compare:
        with open(arguments.compare, encoding='utf8') as handle:
            compare(numeric_data(), json.load(handle))
        print('Python independently matches the published JS quantities')
    else:
        print(json.dumps(numeric_data(), ensure_ascii=False, indent=2))
