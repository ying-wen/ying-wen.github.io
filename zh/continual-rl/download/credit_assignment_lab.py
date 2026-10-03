"""Standard-library experiments for temporal credit assignment.

Run: python3 credit_assignment_lab.py all | frozen | online | control | recurrent | test
Code: MIT. Independent small implementations; no author benchmark is reproduced.
Linear TD uses fixed features and a constant scalar step size within a run.
"""
import argparse
import math
import random
import unittest


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def validate(features, rewards, weights, gamma, lam, alpha=0.1):
    if len(features) != len(rewards) + 1 or not weights:
        raise ValueError('T+1 feature vectors and a non-empty weight vector required')
    if any(len(x) != len(weights) for x in features):
        raise ValueError('feature and weight dimensions must agree')
    values = [gamma, lam, alpha, *weights, *rewards]
    values.extend(v for x in features for v in x)
    if not all(math.isfinite(v) for v in values):
        raise ValueError('finite inputs required')
    if not 0 <= gamma <= 1 or not 0 <= lam <= 1 or alpha < 0:
        raise ValueError('gamma, lambda in [0,1], alpha >= 0 required')


# BEGIN frozen
def lambda_returns(rewards, values, gamma, lam):
    """Finite-horizon forward targets; values[-1] is zero only at true terminal."""
    if len(values) != len(rewards) + 1 or not values:
        raise ValueError('T+1 values required')
    if not 0 <= gamma <= 1 or not 0 <= lam <= 1:
        raise ValueError('gamma and lambda must lie in [0,1]')
    targets = [0.0] * len(rewards)
    tail = values[-1]
    for t in range(len(rewards) - 1, -1, -1):
        tail = rewards[t] + gamma * ((1 - lam) * values[t + 1] + lam * tail)
        targets[t] = tail
    return targets


def frozen_forward_backward(features, rewards, weights, gamma, lam, alpha):
    """Two independently accumulated episode increments using ONE fixed w."""
    validate(features, rewards, weights, gamma, lam, alpha)
    values = [dot(weights, x) for x in features]
    targets = lambda_returns(rewards, values, gamma, lam)
    forward, backward, trace = [[0.0] * len(weights) for _ in range(3)]
    for t, x in enumerate(features[:-1]):
        for j in range(len(weights)):
            forward[j] += alpha * (targets[t] - values[t]) * x[j]
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        trace = [gamma * lam * z + xj for z, xj in zip(trace, x)]
        backward = [b + alpha * delta * z for b, z in zip(backward, trace)]
    return dict(targets=targets, forward=forward, backward=backward)
# END frozen


# BEGIN true_online
def td_episode(features, rewards, initial, gamma, lam, alpha, true_online=True):
    """Online linear TD(lambda), returning weights after EVERY transition.

    Terminal features must be zero. A nonzero last feature denotes a truncated
    continuing segment with its current value used as bootstrap. No artificial
    parameter reset occurs inside the segment.
    """
    validate(features, rewards, initial, gamma, lam, alpha)
    weights = list(initial)
    trace = [0.0] * len(weights)
    old_value = 0.0
    history = [list(weights)]
    for t, reward in enumerate(rewards):
        x, next_x = features[t], features[t + 1]
        value, next_value = dot(weights, x), dot(weights, next_x)
        delta = reward + gamma * next_value - value
        if true_online:
            # The overlap is computed from the OLD trace.
            overlap = dot(trace, x)
            trace = [gamma * lam * z + (1 - alpha * gamma * lam * overlap) * xj
                     for z, xj in zip(trace, x)]
            correction = value - old_value
            weights = [w + alpha * (delta + correction) * z - alpha * correction * xj
                       for w, z, xj in zip(weights, trace, x)]
        else:
            trace = [gamma * lam * z + xj for z, xj in zip(trace, x)]
            weights = [w + alpha * delta * z for w, z in zip(weights, trace)]
        old_value = next_value  # PRE-update prediction, not the new prediction.
        history.append(list(weights))
    return history
# END true_online


# BEGIN online_forward
def online_forward_reference(features, rewards, initial, gamma, lam, alpha):
    """Literal interim lambda-return algorithm; deliberately expensive.

    At each horizon h, recompute all interim targets and replay their sequential
    updates from initial. The n-step bootstrap uses the actual weights at time
    t+n-1, NOT the temporary helper weights used in this replay. This reference
    uses only the prefix observed so far, but stores and recomputes old data.
    """
    validate(features, rewards, initial, gamma, lam, alpha)
    history = [list(initial)]
    for horizon in range(1, len(rewards) + 1):
        helper = list(initial)
        for t in range(horizon):
            target, reward_sum = 0.0, 0.0
            remaining = horizon - t
            for n in range(1, remaining + 1):
                reward_sum += gamma ** (n - 1) * rewards[t + n - 1]
                bootstrap = dot(history[t + n - 1], features[t + n])
                n_step = reward_sum + gamma ** n * bootstrap
                mixture = lam ** (n - 1)
                if n < remaining:
                    mixture *= 1 - lam
                target += mixture * n_step
            error = target - dot(helper, features[t])
            helper = [w + alpha * error * x for w, x in zip(helper, features[t])]
        history.append(helper)
    return history
# END online_forward


# BEGIN control
def control_trace_step(q, trace, state, action, reward, next_state, next_action,
                       gamma=0.9, lam=0.8, alpha=0.1, watkins=False, terminal=False):
    """Tabular accumulating Sarsa(lambda) or Watkins Q(lambda), one transition.

    Input trace is already decayed/cut by the preceding step. next_action must
    be selected using PRE-update q. For Watkins, determine whether it ties for
    the maximum before changing q. Cutting affects FUTURE credit only.
    """
    key = (state, action)
    if key not in q:
        raise ValueError('current state-action pair missing from q')
    greedy_next = False
    if terminal:
        target = reward
    else:
        next_values = [v for (s, _), v in q.items() if s == next_state]
        if not next_values or (next_state, next_action) not in q:
            raise ValueError('next state-action values required')
        greedy_next = q[(next_state, next_action)] == max(next_values)
        bootstrap = max(next_values) if watkins else q[(next_state, next_action)]
        target = reward + gamma * bootstrap
    delta = target - q[key]
    active = {pair: trace.get(pair, 0.0) for pair in q}
    active[key] += 1.0
    updated = {pair: value + alpha * delta * active[pair] for pair, value in q.items()}
    keep = not terminal and (not watkins or greedy_next)
    carry = {pair: gamma * lam * value if keep else 0.0 for pair, value in active.items()}
    return dict(q=updated, trace=carry, delta=delta, greedy_next=greedy_next)
# END control


# BEGIN recurrent
def recurrent_sensitivity(inputs, b, target=1.0, decay=0.5, truncate_after=None):
    """Fixed-parameter h_t=decay*h_(t-1)+b*u_t, and exact dh_t/db.

    truncate_after cuts the derivative before that zero-based input index.
    The numerical hidden state is retained, demonstrating detach vs reset.
    """
    h, sensitivity = 0.0, 0.0
    for t, value in enumerate(inputs):
        if t == truncate_after:
            sensitivity = 0.0
        h = decay * h + b * value
        sensitivity = decay * sensitivity + value
    loss = 0.5 * (h - target) ** 2
    return dict(h=h, sensitivity=sensitivity, loss=loss,
                gradient=(h - target) * sensitivity)
# END recurrent


def frozen_demo():
    result = frozen_forward_backward([[1, 0], [0, 1], [0, 0]], [0, 1],
                                     [0.2, 0.4], 0.9, 0.8, 0.1)
    print('two-state frozen:', result)


def online_demo():
    features, rewards, initial = [[1], [1], [0]], [1, 2], [0]
    args = features, rewards, initial, 1.0, 1.0, 0.5
    print('frozen increment:', frozen_forward_backward(*args)['forward'])
    print('traditional TD:', td_episode(*args, true_online=False))
    print('true-online TD:', td_episode(*args))
    print('online-forward:', online_forward_reference(*args))


def control_demo():
    q = {(0, 0): 0.0, (1, 0): 2.0, (1, 1): 1.0}
    for watkins in (False, True):
        result = control_trace_step(q, {}, 0, 0, 0, 1, 1, watkins=watkins)
        print('Watkins' if watkins else 'Sarsa', result)


def recurrent_demo():
    print('full:', recurrent_sensitivity([1, 0, 0], 0.2))
    print('detach after first input:', recurrent_sensitivity([1, 0, 0], 0.2, truncate_after=1))


class Checks(unittest.TestCase):
    def assert_vector_close(self, a, b, places=10):
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            self.assertAlmostEqual(x, y, places=places)

    def test_lambda_endpoints(self):
        rewards, values = [1, 2], [0.2, 0.4, 0]
        self.assert_vector_close(lambda_returns(rewards, values, 0.9, 0), [1.36, 2])
        self.assert_vector_close(lambda_returns(rewards, values, 0.9, 1), [2.8, 2])

    def test_lambda_nonterminal_tail(self):
        self.assert_vector_close(lambda_returns([1], [0.2, 3], 0.9, 0.8), [3.7])

    def test_forward_backward_hand_calculation(self):
        result = frozen_forward_backward([[1, 0], [0, 1], [0, 0]], [0, 1],
                                         [0.2, 0.4], 0.9, 0.8, 0.1)
        self.assert_vector_close(result['targets'], [0.792, 1])
        self.assert_vector_close(result['forward'], [0.0592, 0.06])
        self.assert_vector_close(result['backward'], result['forward'])

    def test_frozen_equivalence_dense_features(self):
        rng = random.Random(14)
        for lam in (0, 0.2, 0.8, 1):
            features = [[rng.uniform(-1, 1) for _ in range(3)] for _ in range(7)]
            rewards = [rng.uniform(-2, 2) for _ in range(6)]
            result = frozen_forward_backward(features, rewards, [0.2, -0.1, 0.3],
                                             0.93, lam, 0.1)
            self.assert_vector_close(result['forward'], result['backward'])

    def test_true_online_hand_calculation(self):
        args = [[1], [1], [0]], [1, 2], [0], 1, 1, 0.5
        self.assert_vector_close(td_episode(*args)[-1], [1.75])
        self.assert_vector_close(online_forward_reference(*args)[-1], [1.75])
        self.assert_vector_close(td_episode(*args, true_online=False)[-1], [2])
        self.assert_vector_close(frozen_forward_backward(*args)['forward'], [2.5])

    def test_true_online_equals_reference_every_prefix(self):
        rng = random.Random(42)
        for terminal in (False, True):
            for gamma in (0, 0.9, 1):
                for lam in (0, 0.5, 1):
                    features = [[rng.uniform(-1, 1) for _ in range(3)] for _ in range(7)]
                    if terminal:
                        features[-1] = [0, 0, 0]
                    rewards = [rng.uniform(-1, 1) for _ in range(6)]
                    args = features, rewards, [0.3, -0.2, 0.1], gamma, lam, 0.27
                    for actual, expected in zip(td_episode(*args), online_forward_reference(*args)):
                        self.assert_vector_close(actual, expected)

    def test_lambda_zero_reduces_to_td_zero(self):
        args = [[1, 0.5], [0.5, 1], [0, 0]], [1, -0.5], [0.4, -0.1], 0.9, 0, 0.1
        for a, b in zip(td_episode(*args), td_episode(*args, true_online=False)):
            self.assert_vector_close(a, b)

    def test_first_step_with_nonzero_initial_prediction(self):
        args = [[2], [0]], [1], [0.3], 0.9, 0.8, 0.1
        self.assert_vector_close(td_episode(*args)[-1], [0.38])

    def test_terminal_reward_reaches_old_features(self):
        result = td_episode([[1, 0], [0, 1], [0, 0]], [0, 1], [0, 0], 0.9, 0.8, 0.1)
        self.assert_vector_close(result[-1], [0.072, 0.1])

    def test_watkins_cuts_after_current_update(self):
        q = {(0, 0): 0.0, (1, 0): 2.0, (1, 1): 1.0}
        result = control_trace_step(q, {}, 0, 0, 0, 1, 1, watkins=True)
        self.assertAlmostEqual(result['q'][(0, 0)], 0.18)
        self.assertTrue(all(z == 0 for z in result['trace'].values()))
        self.assertEqual(q[(0, 0)], 0)

    def test_watkins_greedy_tie_keeps_trace(self):
        q = {(0, 0): 0.0, (1, 0): 2.0, (1, 1): 2.0}
        result = control_trace_step(q, {}, 0, 0, 0, 1, 1, watkins=True)
        self.assertAlmostEqual(result['trace'][(0, 0)], 0.72)

    def test_sarsa_uses_selected_action(self):
        q = {(0, 0): 0.0, (1, 0): 2.0, (1, 1): 1.0}
        result = control_trace_step(q, {}, 0, 0, 0, 1, 1)
        self.assertAlmostEqual(result['q'][(0, 0)], 0.09)
        self.assertAlmostEqual(result['trace'][(0, 0)], 0.72)

    def test_control_terminal_clears_after_credit(self):
        result = control_trace_step({(0, 0): 0.0, (1, 0): 0.0}, {(0, 0): 0.72},
                                    1, 0, 1, None, None, terminal=True)
        self.assertAlmostEqual(result['q'][(0, 0)], 0.072)
        self.assertTrue(all(z == 0 for z in result['trace'].values()))

    def test_recurrent_finite_difference(self):
        inputs, b, eps = [1, 0, 0], 0.2, 1e-6
        row = recurrent_sensitivity(inputs, b)
        numerical = (recurrent_sensitivity(inputs, b + eps)['loss']
                     - recurrent_sensitivity(inputs, b - eps)['loss']) / (2 * eps)
        self.assertAlmostEqual(row['gradient'], numerical, places=9)
        self.assertAlmostEqual(row['gradient'], -0.2375)

    def test_detach_does_not_erase_hidden_state(self):
        full = recurrent_sensitivity([1, 0, 0], 0.2)
        short = recurrent_sensitivity([1, 0, 0], 0.2, truncate_after=1)
        self.assertAlmostEqual(short['h'], full['h'])
        self.assertEqual(short['gradient'], 0)
        self.assertNotEqual(full['gradient'], 0)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            td_episode([[1]], [1], [0], 0.9, 0.8, 0.1)
        with self.assertRaises(ValueError):
            td_episode([[1], [0]], [1], [0], 0.9, 1.2, 0.1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['all', 'frozen', 'online', 'control', 'recurrent', 'test'])
    command = parser.parse_args().command
    if command == 'test':
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
        raise SystemExit(0 if result.wasSuccessful() else 1)
    for name, demo in [('frozen', frozen_demo), ('online', online_demo),
                       ('control', control_demo), ('recurrent', recurrent_demo)]:
        if command in ('all', name):
            print(name)
            demo()


if __name__ == '__main__':
    main()
