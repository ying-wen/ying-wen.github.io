"""Standard-library experiments for temporal credit assignment.

Run: python3 credit_assignment_lab.py all | frozen | online | control | recurrent
     | offpolicy | adaptive | expected | gradient | actor | test
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


# BEGIN offpolicy
def error_kernel_targets(values, deltas, incoming, gamma):
    """Q targets from fixed errors and incoming c_t; c_0 is never used.

    G_t - Q_t = delta_t + gamma*c_(t+1)*(G_(t+1)-Q_(t+1)).
    This is an independently written finite-path calculation, not a learner.
    """
    n = len(deltas)
    if len(values) != n or len(incoming) != n:
        raise ValueError('equal-length values, errors, and incoming coefficients required')
    if not 0 <= gamma <= 1 or any(c < 0 or not math.isfinite(c) for c in incoming):
        raise ValueError('gamma in [0,1] and finite nonnegative coefficients required')
    targets, correction = [0.0] * n, 0.0
    for t in range(n - 1, -1, -1):
        carry = gamma * incoming[t + 1] if t + 1 < n else 0.0
        correction = deltas[t] + carry * correction
        targets[t] = values[t] + correction
    return targets


def offpolicy_trace_coefficients(target_probs, behavior_probs, lam, method):
    if len(target_probs) != len(behavior_probs) or not 0 <= lam <= 1:
        raise ValueError('matching action probabilities and lambda in [0,1] required')
    coefficients = []
    for pi, mu in zip(target_probs, behavior_probs):
        if not 0 <= pi <= 1 or not 0 < mu <= 1:
            raise ValueError('sampled actions require positive behavior probability')
        ratio = pi / mu
        if method == 'is':
            coefficient = lam * ratio
        elif method == 'tree':
            coefficient = lam * pi
        elif method == 'retrace':
            coefficient = lam * min(1.0, ratio)
        else:
            raise ValueError('method must be is, tree, or retrace')
        coefficients.append(coefficient)
    return coefficients


def q_sigma_targets(q_taken, expected_next, rewards, next_probs, gamma, lam, sigma):
    """Frozen ON-POLICY Q(sigma) with geometric lambda mixing.

    sigma controls sampled/expected actions; lambda controls return length.
    No off-policy Q(sigma) correction is implemented here.
    """
    n = len(rewards)
    if (len(q_taken) != n + 1 or len(expected_next) != n
            or len(next_probs) != n or not 0 <= sigma <= 1 or not 0 <= lam <= 1):
        raise ValueError('T+1 sampled Q and T expectations/probabilities required')
    deltas = [r + gamma * (sigma * q_taken[t + 1] + (1 - sigma) * expected_next[t])
              - q_taken[t] for t, r in enumerate(rewards)]
    incoming = [0.0] + [lam * (sigma + (1 - sigma) * next_probs[t])
                         for t in range(n - 1)]
    return error_kernel_targets(q_taken[:-1], deltas, incoming, gamma)


def vtrace_targets(rewards, values, discounts, ratios, rho_cap=1.0, c_cap=1.0):
    """Finite V-trace target and actor advantage, with explicit tail bootstrap.

    The continuation coefficient c_t is indexed at the CURRENT action, unlike
    the incoming coefficient c_(t+1) in the action-value kernel above.
    rho_cap=None leaves the error correction ratio unclipped.
    """
    n = len(rewards)
    if len(values) != n + 1 or len(discounts) != n or len(ratios) != n:
        raise ValueError('T+1 values and T rewards/discounts/ratios required')
    if c_cap <= 0 or (rho_cap is not None and rho_cap < c_cap):
        raise ValueError('0 < c_cap <= rho_cap required')
    if any(not 0 <= d <= 1 for d in discounts) or any(r < 0 for r in ratios):
        raise ValueError('discounts in [0,1] and nonnegative ratios required')
    clipped = [min(rho_cap, r) if rho_cap is not None else r for r in ratios]
    vs, correction = [0.0] * n, 0.0
    for t in range(n - 1, -1, -1):
        delta = rewards[t] + discounts[t] * values[t + 1] - values[t]
        correction = clipped[t] * delta + discounts[t] * min(c_cap, ratios[t]) * correction
        vs[t] = values[t] + correction
    next_targets = vs[1:] + [values[-1]]
    advantages = [rho * (r + d * tail - value)
                  for rho, r, d, tail, value in zip(clipped, rewards, discounts,
                                                    next_targets, values[:-1])]
    return dict(targets=vs, advantages=advantages)


def clipped_policy(target, behavior, rho_cap):
    """Tabular V-trace fixed-point policy, not the original target policy."""
    if len(target) != len(behavior) or not target or rho_cap <= 0:
        raise ValueError('matching nonempty distributions and positive rho cap required')
    if (any(p < 0 for p in target) or any(p <= 0 for p in behavior)
            or abs(sum(target) - 1) > 1e-10 or abs(sum(behavior) - 1) > 1e-10):
        raise ValueError('normalized distributions with behavior support required')
    masses = [min(pi, rho_cap * mu) for pi, mu in zip(target, behavior)]
    return [mass / sum(masses) for mass in masses]
# END offpolicy


# BEGIN adaptive
def variable_lambda_equivalence(features, rewards, weights, discounts, lambdas):
    """Frozen forward/backward identity with arrival-indexed gamma_t/lambda_t."""
    n = len(rewards)
    if len(discounts) != n + 1 or len(lambdas) != n + 1:
        raise ValueError('T+1 arrival-indexed discounts and lambdas required')
    validate(features, rewards, weights, 1.0, 1.0)
    if any(not 0 <= v <= 1 for v in [*discounts, *lambdas]):
        raise ValueError('discounts and lambdas must lie in [0,1]')
    values = [dot(weights, x) for x in features]
    targets, tail = [0.0] * n, values[-1]
    for t in range(n - 1, -1, -1):
        tail = rewards[t] + discounts[t + 1] * (
            (1 - lambdas[t + 1]) * values[t + 1] + lambdas[t + 1] * tail)
        targets[t] = tail
    forward, backward, trace = [[0.0] * len(weights) for _ in range(3)]
    for t in range(n):
        delta = rewards[t] + discounts[t + 1] * values[t + 1] - values[t]
        trace = [discounts[t] * lambdas[t] * z + x for z, x in zip(trace, features[t])]
        forward = [a + (targets[t] - values[t]) * x for a, x in zip(forward, features[t])]
        backward = [a + delta * z for a, z in zip(backward, trace)]
    return dict(targets=targets, forward=forward, backward=backward)


def greedy_lambda(bias_squared, return_variance):
    """Oracle optimum of the LOCAL bias/variance surrogate, not full lambda-greedy.

    White & White's algorithm must also learn return moments online. We supply
    those statistics as inputs here so the closed-form optimization is testable.
    """
    if any(not math.isfinite(v) or v < 0 for v in (bias_squared, return_variance)):
        raise ValueError('finite nonnegative squared bias and variance required')
    total = bias_squared + return_variance
    return bias_squared / total if total else 0.0
# END adaptive


# BEGIN expected
def expected_trace_enumeration(hidden_history=False):
    """Exact finite enumeration at a merging state, including an aliasing failure.

    Two equiprobable histories have traces (.72,0,1) and (0,.72,1).
    Markov case: reward in {0,2} is independent of the incoming history.
    Aliased case: the hidden incoming history determines reward 2 vs 0.
    Predictions and features are frozen; this is not an ET training benchmark.
    """
    traces = [[0.72, 0.0, 1.0], [0.0, 0.72, 1.0]]
    expected_trace = [sum(z[j] for z in traces) / 2 for j in range(3)]
    cases = [(0.5, 2.0, traces[0]), (0.5, 0.0, traces[1])] if hidden_history else [
        (0.25, reward, trace) for trace in traces for reward in (0.0, 2.0)]
    result = {}
    for name, use_expected in [('instantaneous', False), ('expected', True)]:
        updates = [(prob, [reward * z for z in (expected_trace if use_expected else trace)])
                   for prob, reward, trace in cases]
        mean = [sum(prob * row[j] for prob, row in updates) for j in range(3)]
        variance = [sum(prob * (row[j] - mean[j]) ** 2 for prob, row in updates)
                    for j in range(3)]
        result[name] = dict(mean=mean, variance=variance)
    return result
# END expected


def nonlinear_value_gradient(weights, x):
    value = math.tanh(dot(weights, x))
    return value, [(1 - value * value) * feature for feature in x]


# BEGIN gradient
def gradient_trace_equivalence(features, rewards, weights, auxiliary, gamma, lam,
                               correction=False, regularization=0.0):
    """Frozen nonlinear GTD2/TDC/TDRC total-increment identities, on-policy.

    V=tanh(w dot x), H=tanh(theta dot x). Independently differentiate the
    recursive lambda target for the forward view, then accumulate the backward
    scalar-H trace and parameter traces. No parameters change during this call.
    The saddle objective is half-scaled: sum(H*delta_lambda - H**2/2).
    This is a mathematical unit experiment, not the authors' deep RL benchmark.
    """
    validate(features, rewards, weights, gamma, lam)
    if len(auxiliary) != len(weights) or regularization < 0:
        raise ValueError('matching auxiliary dimension and nonnegative penalty required')
    n, d = len(rewards), len(weights)
    value_rows = [nonlinear_value_gradient(weights, x) for x in features]
    values, gradients = zip(*value_rows)
    h_rows = [nonlinear_value_gradient(auxiliary, x) for x in features[:-1]]
    targets, target_gradients = [0.0] * n, [[0.0] * d for _ in range(n)]
    tail, tail_gradient = values[-1], list(gradients[-1])
    for t in range(n - 1, -1, -1):
        tail = rewards[t] + gamma * ((1 - lam) * values[t + 1] + lam * tail)
        tail_gradient = [gamma * ((1 - lam) * g + lam * old)
                         for g, old in zip(gradients[t + 1], tail_gradient)]
        targets[t], target_gradients[t] = tail, list(tail_gradient)
    fw, fh, bw, bh, trace_v, trace_h = [[0.0] * d for _ in range(6)]
    scalar_trace, objective = 0.0, 0.0
    for t, (h, grad_h) in enumerate(h_rows):
        grad_v = gradients[t]
        error = targets[t] - values[t]
        grad_error = [g - v for g, v in zip(target_gradients[t], grad_v)]
        objective += h * error - 0.5 * h * h - 0.5 * regularization * dot(auxiliary, auxiliary)
        fw = [a - h * ge + ((error - h) * gv if correction else 0.0)
              for a, ge, gv in zip(fw, grad_error, grad_v)]
        fh = [a + (error - h) * gh - regularization * theta
              for a, gh, theta in zip(fh, grad_h, auxiliary)]
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        grad_delta = [gamma * gn - gv for gn, gv in zip(gradients[t + 1], grad_v)]
        scalar_trace = gamma * lam * scalar_trace + h
        trace_h = [gamma * lam * z + g for z, g in zip(trace_h, grad_h)]
        trace_v = [gamma * lam * z + g for z, g in zip(trace_v, grad_v)]
        bw = [a - scalar_trace * gd + (delta * zv - h * gv if correction else 0.0)
              for a, gd, zv, gv in zip(bw, grad_delta, trace_v, grad_v)]
        bh = [a + delta * zh - h * gh - regularization * theta
              for a, zh, gh, theta in zip(bh, trace_h, grad_h, auxiliary)]
    return dict(forward_w=fw, backward_w=bw, forward_h=fh, backward_h=bh,
                objective=objective)
# END gradient


# BEGIN actor
def actor_trace_equivalence(scores, rewards, values, gamma, lam):
    """Frozen discounted-start-state policy scores; actor is not updated here."""
    if len(scores) != len(rewards) or not scores or len(values) != len(rewards) + 1:
        raise ValueError('T score vectors/rewards and T+1 values required')
    d = len(scores[0])
    if not d or any(len(score) != d for score in scores):
        raise ValueError('score dimensions must match')
    advantages = [g - v for g, v in zip(lambda_returns(rewards, values, gamma, lam), values)]
    forward, backward, trace = [[0.0] * d for _ in range(3)]
    for t, score in enumerate(scores):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        trace = [gamma * lam * z + gamma ** t * s for z, s in zip(trace, score)]
        forward = [a + gamma ** t * advantages[t] * s for a, s in zip(forward, score)]
        backward = [a + delta * z for a, z in zip(backward, trace)]
    return dict(forward=forward, backward=backward, advantages=advantages)
# END actor


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


def offpolicy_demo():
    for method in ('is', 'tree', 'retrace'):
        coefficients = offpolicy_trace_coefficients([0.4, 0.8, 0.2], [0.4, 0.2, 0.8], 0.8, method)
        print(method, 'c:', coefficients, 'targets:',
              error_kernel_targets([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], coefficients, 0.9))
    for sigma in (0.0, 0.5, 1.0):
        print('Q(sigma), sigma=', sigma, q_sigma_targets([0.0, 0.4, 0.0],
              [0.2, 0.0], [0.0, 1.0], [0.5, 0.0], 0.9, 0.8, sigma))
    print('V-trace:', vtrace_targets([0.0, 1.0], [0.2, 0.4, 0.0], [0.9, 0.0], [0.5, 2.0]))
    print('clipped target policy:', clipped_policy([0.2, 0.8], [0.8, 0.2], 1.0))


def adaptive_demo():
    print('variable lambda:', variable_lambda_equivalence([[1, 0], [0, 1], [1, 1]],
          [0, 1], [0.2, 0.4], [1.0, 0.9, 0.0], [0.3, 0.8, 0.5]))
    print('local optimal lambda:', greedy_lambda(4.0, 1.0))


def expected_demo():
    print('Markov merging state:', expected_trace_enumeration())
    print('aliased merging observation:', expected_trace_enumeration(hidden_history=True))


def gradient_demo():
    args = [[1.0, 0.5], [0.2, 1.0], [0.0, 0.0]], [0.0, 1.0], [0.2, -0.1], [0.3, 0.1], 0.9, 0.8
    for name, correction, penalty in [('GTD2', False, 0.0), ('TDC', True, 0.0), ('TDRC', True, 0.1)]:
        result = gradient_trace_equivalence(*args, correction=correction, regularization=penalty)
        max_error = max(abs(a - b) for field in ('w', 'h')
                        for a, b in zip(result['forward_' + field], result['backward_' + field]))
        print(name, 'frozen total-increment discrepancy:', max_error, result)


def actor_demo():
    print('discounted actor:', actor_trace_equivalence([[0.6], [-0.7]], [0, 1], [0.2, 0.4, 0], 0.9, 0.8))


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

    def test_offpolicy_kernel_equals_explicit_sum_and_backward_trace(self):
        gamma, deltas, coefficients = 0.9, [0.2, -0.3, 1.0], [99.0, 0.8, 0.2]
        features = [[1, 0], [0.5, 1], [0, 1]]
        targets = error_kernel_targets([0, 0, 0], deltas, coefficients, gamma)
        for t in range(3):
            explicit, product = 0.0, 1.0
            for k in range(t, 3):
                if k > t:
                    product *= gamma * coefficients[k]
                explicit += product * deltas[k]
            self.assertAlmostEqual(targets[t], explicit)
        forward = [sum(g * x[j] for g, x in zip(targets, features)) for j in range(2)]
        trace, backward = [0, 0], [0, 0]
        for t in range(3):
            trace = [gamma * coefficients[t] * z + x for z, x in zip(trace, features[t])]
            backward = [b + deltas[t] * z for b, z in zip(backward, trace)]
        self.assert_vector_close(forward, backward)

    def test_retrace_near_onpolicy_keeps_more_than_tree_backup(self):
        pi = mu = [0.1, 0.4]
        self.assert_vector_close(offpolicy_trace_coefficients(pi, mu, 0.8, 'retrace'), [0.8, 0.8])
        self.assert_vector_close(offpolicy_trace_coefficients(pi, mu, 0.8, 'tree'), [0.08, 0.32])

    def test_importance_ratio_can_amplify_trace(self):
        self.assertEqual(offpolicy_trace_coefficients([0.8], [0.01], 0.8, 'is'), [64.0])
        self.assertEqual(offpolicy_trace_coefficients([0.8], [0.01], 0.8, 'retrace'), [0.8])
        with self.assertRaises(ValueError):
            offpolicy_trace_coefficients([0.8], [0.0], 0.8, 'retrace')

    def test_q_sigma_endpoints(self):
        q, rewards, gamma, lam = [0.1, 0.4, 0.0], [0.0, 1.0], 0.9, 0.8
        sarsa = q_sigma_targets(q, [0.2, 0.0], rewards, [0.5, 0.0], gamma, lam, 1.0)
        self.assert_vector_close(sarsa, lambda_returns(rewards, q, gamma, lam))
        expected_errors = [0.9 * 0.2 - 0.1, 1.0 - 0.4]
        tree = error_kernel_targets(q[:-1], expected_errors, [0, 0.8 * 0.5], gamma)
        self.assert_vector_close(q_sigma_targets(q, [0.2, 0], rewards, [0.5, 0], gamma, lam, 0), tree)

    def test_vtrace_onpolicy_equals_full_return_with_bootstrap(self):
        rewards, values = [1.0, 2.0], [0.2, 0.4, 3.0]
        result = vtrace_targets(rewards, values, [0.9, 0.9], [1.0, 1.0])
        self.assert_vector_close(result['targets'], lambda_returns(rewards, values, 0.9, 1.0))
        self.assert_vector_close(result['advantages'], [5.23 - 0.2, 4.7 - 0.4])

    def test_vtrace_matches_explicit_kernel_current_ratio_index(self):
        rewards, values, discounts, ratios = [0.0, 1.0], [0.2, 0.4, 0.0], [0.9, 0.0], [0.5, 2.0]
        result = vtrace_targets(rewards, values, discounts, ratios)
        # D_1 = 1*.6, D_0 = .5*.16 + .9*.5*.6 = .35.
        self.assert_vector_close(result['targets'], [0.55, 1.0])
        self.assert_vector_close(result['advantages'], [0.35, 0.6])

    def test_vtrace_rho_cap_changes_bandit_fixed_point(self):
        pi, mu = [0.2, 0.8], [0.8, 0.2]
        mixed = clipped_policy(pi, mu, 1.0)
        self.assert_vector_close(mixed, [0.5, 0.5])
        for cap, fixed in [(1.0, 0.5), (4.0, 0.8)]:
            expected_delta = sum(m * min(cap, p / m) * (r - fixed)
                                 for p, m, r in zip(pi, mu, [0.0, 1.0]))
            self.assertAlmostEqual(expected_delta, 0.0)

    def test_state_lambda_arrival_index_and_terminal_reward(self):
        result = variable_lambda_equivalence([[1, 0], [0, 1], [1, 1]], [0, 1],
                                             [0.2, 0.4], [1, 0.9, 0], [0.1, 0.8, 0.2])
        self.assert_vector_close(result['targets'], [0.792, 1])
        self.assert_vector_close(result['forward'], [0.592, 0.6])
        self.assert_vector_close(result['backward'], result['forward'])

    def test_state_lambda_nonterminal_dense_equivalence(self):
        result = variable_lambda_equivalence([[1, 0.4], [0.3, 1], [0.1, 0.2], [1, -0.3]],
          [0.2, -0.7, 1], [0.4, -0.2], [1, 0.8, 0.6, 0.9], [0.9, 0.2, 0.8, 0.5])
        self.assert_vector_close(result['forward'], result['backward'])

    def test_greedy_lambda_is_local_mse_minimum(self):
        b, variance = 4.0, 1.0
        optimum = greedy_lambda(b, variance)
        self.assertAlmostEqual(optimum, 0.8)
        objective = lambda x: (1 - x) ** 2 * b + x * x * variance
        self.assertLessEqual(objective(optimum), min(objective(i / 1000) for i in range(1001)))
        self.assertEqual(greedy_lambda(0, 1), 0)
        self.assertEqual(greedy_lambda(1, 0), 1)
        self.assertEqual(greedy_lambda(0, 0), 0)

    def test_expected_trace_markov_mean_and_component_variance(self):
        result = expected_trace_enumeration()
        ordinary, expected = result['instantaneous'], result['expected']
        self.assert_vector_close(ordinary['mean'], expected['mean'])
        self.assert_vector_close(ordinary['variance'], [0.3888, 0.3888, 1])
        self.assert_vector_close(expected['variance'], [0.1296, 0.1296, 1])

    def test_expected_trace_aliasing_changes_mean(self):
        result = expected_trace_enumeration(hidden_history=True)
        self.assert_vector_close(result['instantaneous']['mean'], [0.72, 0.0, 1.0])
        self.assert_vector_close(result['expected']['mean'], [0.36, 0.36, 1.0])

    def test_nonlinear_gradient_traces_frozen_total_equivalence(self):
        rng = random.Random(19)
        for terminal in (False, True):
            features = [[rng.uniform(-1, 1) for _ in range(3)] for _ in range(7)]
            if terminal:
                features[-1] = [0.0] * 3
            for lam in (0.0, 0.3, 1.0):
                for correction in (False, True):
                    result = gradient_trace_equivalence(features, [0.1, -0.2, 1, 0.4, 0, 0.6],
                      [0.2, -0.3, 0.4], [0.3, 0.4, -0.1], 0.93, lam, correction, 0.1)
                    self.assert_vector_close(result['forward_w'], result['backward_w'])
                    self.assert_vector_close(result['forward_h'], result['backward_h'])

    def test_gtd2_saddle_gradients_finite_difference(self):
        features, rewards = [[1.0, 0.5], [0.2, 1.0], [0.0, 0.0]], [0.0, 1.0]
        weights, auxiliary, eps = [0.2, -0.1], [0.3, 0.1], 1e-6
        result = gradient_trace_equivalence(features, rewards, weights, auxiliary, 0.9, 0.8,
                                            regularization=0.1)
        for field, parameters in [('w', weights), ('h', auxiliary)]:
            for j in range(2):
                positive, negative = list(parameters), list(parameters)
                positive[j] += eps
                negative[j] -= eps
                def objective(p):
                    w, h = (p, auxiliary) if field == 'w' else (weights, p)
                    return gradient_trace_equivalence(features, rewards, w, h, 0.9, 0.8,
                                                       regularization=0.1)['objective']
                numerical = (objective(positive) - objective(negative)) / (2 * eps)
                direction = -numerical if field == 'w' else numerical
                self.assertAlmostEqual(result['backward_' + field][j], direction, places=9)

    def test_actor_fixed_score_forward_backward_discounted_start_objective(self):
        for lam in (0, 0.8, 1):
            result = actor_trace_equivalence([[0.6, -0.2], [-0.7, 0.1], [0.2, 0.4]],
                                             [0, 0, 1], [0.2, 0.4, 0.1, 0], 0.9, lam)
            self.assert_vector_close(result['forward'], result['backward'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['all', 'frozen', 'online', 'control', 'recurrent',
                                          'offpolicy', 'adaptive', 'expected', 'gradient', 'actor', 'test'])
    command = parser.parse_args().command
    if command == 'test':
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
        raise SystemExit(0 if result.wasSuccessful() else 1)
    for name, demo in [('frozen', frozen_demo), ('online', online_demo),
                       ('control', control_demo), ('recurrent', recurrent_demo),
                       ('offpolicy', offpolicy_demo), ('adaptive', adaptive_demo),
                       ('expected', expected_demo), ('gradient', gradient_demo), ('actor', actor_demo)]:
        if command in ('all', name):
            print(name)
            demo()


if __name__ == '__main__':
    main()
