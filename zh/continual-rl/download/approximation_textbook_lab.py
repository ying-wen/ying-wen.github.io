#!/usr/bin/env python3
"""Small, exact Part II teaching experiments. Python standard library only.

Run: python3 approximation_textbook_lab.py test
     python3 approximation_textbook_lab.py demo
These kernels verify equations, not large-scale algorithm performance.
"""
import argparse
import itertools
import json
import math
import unittest


def dot(x, y):
    if len(x) != len(y):
        raise ValueError("vector dimensions differ")
    return sum(a * b for a, b in zip(x, y))


def solve(matrix, rhs):
    """Gaussian elimination with pivoting, for small nonsingular systems."""
    n = len(rhs)
    if len(matrix) != n or any(len(row) != n for row in matrix):
        raise ValueError("expected a square system")
    rows = [list(row) + [b] for row, b in zip(matrix, rhs)]
    for j in range(n):
        k = max(range(j, n), key=lambda i: abs(rows[i][j]))
        if abs(rows[k][j]) < 1e-12:
            raise ValueError("singular system; use independent features")
        rows[j], rows[k] = rows[k], rows[j]
        pivot = rows[j][j]
        rows[j] = [v / pivot for v in rows[j]]
        for i in range(n):
            if i != j:
                factor = rows[i][j]
                rows[i] = [a - factor * b for a, b in zip(rows[i], rows[j])]
    return [row[-1] for row in rows]


# BEGIN prediction
def gradient_mc(weights, features, target, alpha):
    error = target - dot(weights, features)
    return [w + alpha * error * x for w, x in zip(weights, features)]


def semi_gradient_td(weights, features, reward, next_features, gamma, alpha):
    # Both predictions use the pre-update weights.
    delta = reward + gamma * dot(weights, next_features) - dot(weights, features)
    return [w + alpha * delta * x for w, x in zip(weights, features)], delta


def lstd(samples, dimension):
    """samples = (mass, x, r, gamma_next, x_next); mass may be 1 or probability."""
    a = [[0.0] * dimension for _ in range(dimension)]
    b = [0.0] * dimension
    for mass, x, reward, gamma, xp in samples:
        for i in range(dimension):
            b[i] += mass * reward * x[i]
            for j in range(dimension):
                a[i][j] += mass * x[i] * (x[j] - gamma * xp[j])
    return solve(a, b), a, b


def prediction_demo():
    # A -> B (r=0), B -> A (r=1), gamma=.5, stationary masses .5/.5.
    samples = [(0.5, [1.0], 0.0, 0.5, [3.0]),
               (0.5, [3.0], 1.0, 0.5, [1.0])]
    exact = solve([[1.0, -0.5], [-0.5, 1.0]], [0.0, 1.0])
    td, a, b = lstd(samples, 1)
    # Changing an evaluation weight alone does not change the original stream.
    # To solve the new projected equation, change the actual sample masses.
    weighted, weighted_a, weighted_b = lstd([
        (0.9, [1.0], 0.0, 0.5, [3.0]),
        (0.1, [3.0], 1.0, 0.5, [1.0])], 1)
    mc = (0.5 * exact[0] + 1.5 * exact[1]) / 5.0
    return {"true_values": exact, "A": a, "b": b, "TD_weight": td[0],
            "MC_weight": mc, "TD_values": [td[0], 3 * td[0]],
            "MC_values": [mc, 3 * mc],
            "reweighted_TD_weight": weighted[0],
            "reweighted_A": weighted_a, "reweighted_b": weighted_b}
# END prediction


# BEGIN features_control
def tiles_1d(state, width=0.25, tilings=4):
    """Unhashed sparse keys: one active interval in each offset tiling."""
    if width <= 0 or tilings < 1 or not math.isfinite(state):
        raise ValueError("finite state, positive width and tilings required")
    return tuple((k, math.floor(state / width + k / tilings))
                 for k in range(tilings))


def fourier_1d(state, order=3):
    if not 0 <= state <= 1 or order < 0:
        raise ValueError("normalize state to [0, 1], use nonnegative order")
    return [math.cos(math.pi * k * state) for k in range(order + 1)]


def action_features(features, action, actions):
    if not 0 <= action < actions:
        raise ValueError("action outside action set")
    return [v if a == action else 0.0 for a in range(actions) for v in features]


def semi_gradient_sarsa(weights, x, reward, xp, gamma, alpha, terminated=False):
    # x and xp are state-action features. xp belongs to the already sampled A'.
    if terminated:
        xp = [0.0] * len(weights)
    return semi_gradient_td(weights, x, reward, xp, gamma, alpha)


def features_demo():
    x = action_features([1.0, 0.5], 0, 2)
    xp = action_features([1.0, 1.0], 1, 2)
    weights, delta = semi_gradient_sarsa([0.2, 0.4, 0.1, 0.2], x, 1, xp, 0.9, 0.1)
    return {"tiles_at_0.2": tiles_1d(0.2), "tiles_at_0.21": tiles_1d(0.21),
            "Fourier_at_0.5": fourier_1d(0.5), "delta": delta,
            "updated_weights": weights}
# END features_control


# BEGIN average_control
def differential_sarsa(weights, rate, x, reward, xp, alpha, beta):
    delta = reward - rate + dot(weights, xp) - dot(weights, x)
    # One old delta drives BOTH updates. There is no terminal discount mask.
    next_weights = [w + alpha * delta * value for w, value in zip(weights, x)]
    next_rate = rate + beta * delta
    return next_weights, next_rate, delta


def average_demo():
    # A -> B gives 0; B -> A gives 2. Exact gain=1, h(B)-h(A)=1.
    weights, rate = [0.0, 0.0], 0.0
    for t in range(12000):
        s = t % 2
        x, xp = [0.0, 0.0], [0.0, 0.0]
        x[s], xp[1 - s] = 1.0, 1.0
        weights, rate, _ = differential_sarsa(weights, rate, x, 2.0 * s, xp, 0.05, 0.01)
    return {"gain": rate, "relative_values": [0.0, weights[1] - weights[0]],
            "exact_gain": 1.0, "exact_relative_values": [0.0, 1.0]}
# END average_control


# BEGIN off_policy
def gradient_td(weights, auxiliary, x, reward, xp, gamma, rho, alpha, beta, method):
    if method not in ("GTD2", "TDC") or rho < 0:
        raise ValueError("method must be GTD2/TDC and rho nonnegative")
    delta = reward + gamma * dot(weights, xp) - dot(weights, x)
    hx = dot(auxiliary, x)
    if method == "GTD2":
        direction = [rho * (v - gamma * vp) * hx for v, vp in zip(x, xp)]
    else:
        direction = [rho * (delta * v - gamma * vp * hx) for v, vp in zip(x, xp)]
    next_weights = [w + alpha * d for w, d in zip(weights, direction)]
    # rho multiplies delta, NOT the covariance term hx.
    next_auxiliary = [h + beta * (rho * delta - hx) * v for h, v in zip(auxiliary, x)]
    return next_weights, next_auxiliary


def emphatic_td(weights, trace, followon, previous_rho, x, reward, xp,
                gamma_current, gamma_next, rho, interest, lam, alpha):
    followon = interest + gamma_current * previous_rho * followon
    emphasis = lam * interest + (1 - lam) * followon
    trace = [rho * (gamma_current * lam * e + emphasis * v) for e, v in zip(trace, x)]
    delta = reward + gamma_next * dot(weights, xp) - dot(weights, x)
    weights = [w + alpha * delta * e for w, e in zip(weights, trace)]
    return weights, trace, followon, rho


def off_policy_demo():
    # b(go-to-B)=.1 in each state, pi(go-to-B)=1. d_b=(.9,.1).
    # Features x(A)=1, x(B)=2. All rewards zero. A=-.68, C=1.3.
    samples = [(0.81, [1.0], [1.0], 0.0), (0.09, [1.0], [2.0], 10.0),
               (0.09, [2.0], [1.0], 0.0), (0.01, [2.0], [2.0], 10.0)]
    w, h, td = [1.0], [0.0], 1.0
    for _ in range(2000):
        increments_w, increments_h = 0.0, 0.0
        for probability, x, xp, rho in samples:
            wn, hn = gradient_td(w, h, x, 0, xp, 0.9, rho, 0.01, 0.05, "GTD2")
            increments_w += probability * (wn[0] - w[0])
            increments_h += probability * (hn[0] - h[0])
        w, h = [w[0] + increments_w], [h[0] + increments_h]
        td += 0.01 * 0.68 * td
    return {"A": -0.68, "C": 1.3, "expected_TD_after_2000": td,
            "expected_GTD2_after_2000": w[0],
            "sampling": "exact probability-weighted updates, not a stochastic benchmark"}
# END off_policy


# BEGIN traces
def true_online_td(features, rewards, alpha=0.1, gamma=0.9, lam=0.8, initial=None):
    """One episode/prefix. features has len(rewards)+1; terminal features are zero."""
    if len(features) != len(rewards) + 1 or not 0 <= lam <= 1:
        raise ValueError("invalid trajectory or lambda")
    weights = list(initial) if initial is not None else [0.0] * len(features[0])
    trace, old_value, history = [0.0] * len(weights), 0.0, [weights[:]]
    for x, reward, xp in zip(features, rewards, features[1:]):
        value, next_value = dot(weights, x), dot(weights, xp)
        delta = reward + gamma * next_value - value
        correction = 1 - alpha * gamma * lam * dot(trace, x)
        trace = [gamma * lam * e + correction * v for e, v in zip(trace, x)]
        weights = [w + alpha * (delta + value - old_value) * e
                   - alpha * (value - old_value) * v
                   for w, e, v in zip(weights, trace, x)]
        old_value = next_value  # Save the PRE-update prediction at xp.
        history.append(weights[:])
    return history


def online_forward_view(features, rewards, alpha=0.1, gamma=0.9, lam=0.8, initial=None):
    """Slow independent reference: recompute all interim lambda-return updates."""
    initial = list(initial) if initial is not None else [0.0] * len(features[0])
    history = [initial[:]]
    for horizon in range(1, len(rewards) + 1):
        weights = initial[:]
        for start in range(horizon):
            target = 0.0
            for n in range(1, horizon - start + 1):
                end = start + n
                nreturn = sum(gamma ** j * rewards[start + j] for j in range(n))
                nreturn += gamma ** n * dot(history[end - 1], features[end])
                mixture = lam ** (n - 1)
                if end < horizon:
                    mixture *= 1 - lam
                target += mixture * nreturn
            weights = gradient_mc(weights, features[start], target, alpha)
        history.append(weights)
    return history


def traces_demo():
    features, rewards = [[1.0], [1.0], [0.0]], [0.0, 1.0]
    online = true_online_td(features, rewards, alpha=0.5, gamma=1.0, lam=1.0)
    forward = online_forward_view(features, rewards, alpha=0.5, gamma=1.0, lam=1.0)
    return {"true_online_history": online, "forward_history": forward,
            "accumulating_final": 1.0, "true_online_final": online[-1][0]}
# END traces


# BEGIN policy_gradient
def gaussian_scores(action, mean, log_std):
    """Scores with respect to mean and log standard deviation, before transforms."""
    inverse_variance = math.exp(-2 * log_std)
    error = action - mean
    return error * inverse_variance, error * error * inverse_variance - 1


def sigmoid(theta):
    if theta >= 0:
        return 1 / (1 + math.exp(-theta))
    value = math.exp(theta)
    return value / (1 + value)


def reinforce_gradient(actions, rewards, probability, gamma, baseline=0.0):
    """Bernoulli policy with shared scalar logit. Return gradient, not parameter update."""
    if len(actions) != len(rewards) or not 0 < probability < 1:
        raise ValueError("trajectory sizes or policy support invalid")
    result, return_to_go = 0.0, 0.0
    for t in reversed(range(len(rewards))):
        return_to_go = rewards[t] + gamma * return_to_go
        score = actions[t] - probability
        result += gamma ** t * (return_to_go - baseline) * score
    return result


def exact_policy_gradient(theta, gamma=0.5, baseline=0.0):
    # Two decisions, reward 1 iff first action=1 and second action=0.
    # J(theta)=gamma*p*(1-p); enumerate all four trajectories exactly.
    probability, expectation = sigmoid(theta), 0.0
    for actions in itertools.product((0, 1), repeat=2):
        mass = math.prod(probability if a else 1 - probability for a in actions)
        rewards = [0.0, float(actions == (1, 0))]
        expectation += mass * reinforce_gradient(actions, rewards, probability, gamma, baseline)
    return expectation


def policy_demo():
    theta, gamma, eps = 0.7, 0.5, 1e-5
    objective = lambda z: gamma * sigmoid(z) * (1 - sigmoid(z))
    p = sigmoid(theta)
    return {"probability": p, "J": objective(theta),
            "enumerated_gradient": exact_policy_gradient(theta, gamma),
            "analytic_gradient": gamma * p * (1 - p) * (1 - 2 * p),
            "finite_difference": (objective(theta + eps) - objective(theta - eps)) / (2 * eps),
            "with_baseline_3": exact_policy_gradient(theta, gamma, baseline=3.0),
            "bandit_Fisher": p * (1 - p), "bandit_natural_gradient_reward_gap_1": 1.0}
# END policy_gradient


class ApproximationTests(unittest.TestCase):
    def assertVector(self, actual, expected, places=11):
        self.assertEqual(len(actual), len(expected))
        for x, y in zip(actual, expected):
            self.assertAlmostEqual(x, y, places=places)

    def test_linear_system(self):
        self.assertVector(solve([[0, 2], [1, 3]], [4, 7]), [1, 2])

    def test_singular_system(self):
        with self.assertRaises(ValueError):
            solve([[1, 2], [2, 4]], [1, 2])

    def test_vector_dimensions(self):
        with self.assertRaises(ValueError):
            dot([1], [1, 2])

    def test_mc_gradient_finite_difference(self):
        x, weights, target, eps = [1, 2], [0.2, -0.1], 0.7, 1e-6
        update = gradient_mc(weights, x, target, 1.0)
        for j in range(2):
            wp, wm = weights[:], weights[:]
            wp[j] += eps
            wm[j] -= eps
            grad = ((target - dot(wp, x)) ** 2 - (target - dot(wm, x)) ** 2) / (4 * eps)
            self.assertAlmostEqual(update[j] - weights[j], -grad)

    def test_td_old_prediction(self):
        weights, delta = semi_gradient_td([0.5], [2], 1, [3], 0.9, 0.1)
        self.assertAlmostEqual(delta, 1.35)
        self.assertVector(weights, [0.77])

    def test_td_zero_discount_is_regression(self):
        weights, _ = semi_gradient_td([0.5], [2], 1, [100], 0, 0.1)
        self.assertVector(weights, gradient_mc([0.5], [2], 1, 0.1))

    def test_projection_vs_fixed_point(self):
        result = prediction_demo()
        self.assertAlmostEqual(result["TD_weight"], 3 / 7)
        self.assertAlmostEqual(result["MC_weight"], 7 / 15)

    def test_lstd_normal_equation(self):
        result = prediction_demo()
        self.assertAlmostEqual(result["A"][0][0] * result["TD_weight"], result["b"][0])

    def test_td_projection_weights_must_match_sampling(self):
        result = prediction_demo()
        self.assertAlmostEqual(result["reweighted_TD_weight"], 1.0)
        self.assertAlmostEqual(result["reweighted_A"][0][0], 0.3)
        self.assertAlmostEqual(result["reweighted_b"][0], 0.3)
        # The unweighted stream's root does not solve the reweighted equation.
        residual = result["reweighted_b"][0] - result["reweighted_A"][0][0] * result["TD_weight"]
        self.assertAlmostEqual(residual, 6/35)

    def test_scaling_requires_squared_step_adjustment(self):
        w, _ = semi_gradient_td([0.4], [2], 0.3, [1], 0.8, 0.1)
        scaled, _ = semi_gradient_td([0.2], [4], 0.3, [2], 0.8, 0.025)
        self.assertAlmostEqual(2 * scaled[0], w[0])

    def test_tile_count(self):
        self.assertEqual(len(set(tiles_1d(0.2))), 4)

    def test_tile_local_generalization(self):
        self.assertEqual(tiles_1d(0.2), tiles_1d(0.21))

    def test_tile_negative_state(self):
        self.assertEqual(tiles_1d(-0.01, 1, 1), ((0, -1),))

    def test_fourier_endpoints(self):
        self.assertVector(fourier_1d(0, 2), [1, 1, 1])
        self.assertVector(fourier_1d(1, 2), [1, -1, 1])

    def test_invalid_features(self):
        with self.assertRaises(ValueError):
            fourier_1d(2)
        with self.assertRaises(ValueError):
            tiles_1d(0.1, width=0)

    def test_action_blocks(self):
        self.assertVector(action_features([1, 2], 1, 2), [0, 0, 1, 2])

    def test_sarsa_terminal_ignores_bootstrap(self):
        weights, delta = semi_gradient_sarsa([2], [1], 1, [100], 0.9, 0.1, True)
        self.assertAlmostEqual(delta, -1)
        self.assertVector(weights, [1.9])

    def test_differential_simultaneous_update(self):
        w, gain, delta = differential_sarsa([0, 1], 0.5, [1, 0], 0, [0, 1], 0.1, 0.2)
        self.assertEqual(delta, 0.5)
        self.assertVector(w, [0.05, 1])
        self.assertAlmostEqual(gain, 0.6)

    def test_average_additive_value_invariance(self):
        a = differential_sarsa([0, 1], 0.5, [1, 0], 0, [0, 1], 0.1, 0.2)
        b = differential_sarsa([5, 6], 0.5, [1, 0], 0, [0, 1], 0.1, 0.2)
        self.assertAlmostEqual(a[2], b[2])

    def test_average_cycle(self):
        result = average_demo()
        self.assertAlmostEqual(result["gain"], 1.0, places=8)
        self.assertVector(result["relative_values"], [0, 1], places=8)

    def test_gtd2_uses_old_auxiliary(self):
        w, h = gradient_td([1], [0], [1], 0, [2], 0.9, 2, 0.1, 0.2, "GTD2")
        self.assertVector(w, [1])
        self.assertVector(h, [0.32])

    def test_tdc_has_direct_td_term(self):
        w, h = gradient_td([1], [0], [1], 0, [2], 0.9, 2, 0.1, 0.2, "TDC")
        self.assertVector(w, [1.16])
        self.assertVector(h, [0.32])

    def test_zero_ratio_auxiliary_covariance_remains(self):
        w, h = gradient_td([1], [2], [1], 0, [2], 0.9, 0, 0.1, 0.2, "GTD2")
        self.assertVector(w, [1])
        self.assertVector(h, [1.6])

    def test_mspbe_gradient(self):
        a, c, w, eps = -0.68, 1.3, 0.7, 1e-6
        objective = lambda z: (a * z) ** 2 / (2 * c)
        numerical = (objective(w + eps) - objective(w - eps)) / (2 * eps)
        self.assertAlmostEqual(numerical, a * a * w / c)

    def test_emphasis_uses_previous_ratio(self):
        _, trace, followon, rho = emphatic_td([0], [0], 2, 3, [1], 0, [1],
                                               0.5, 0.9, 4, 1, 0, 0.1)
        self.assertEqual(followon, 4)
        self.assertEqual(trace, [16])
        self.assertEqual(rho, 4)

    def test_lambda_one_emphasis_is_interest(self):
        _, trace, _, _ = emphatic_td([0], [2], 20, 3, [1], 0, [1],
                                     0.5, 0.9, 4, 1, 1, 0.1)
        self.assertEqual(trace, [8])

    def test_offpolicy_counterexample(self):
        result = off_policy_demo()
        self.assertGreater(result["expected_TD_after_2000"], 1000)
        self.assertLess(abs(result["expected_GTD2_after_2000"]), 0.01)

    def test_true_online_zero_lambda(self):
        xs, rs, initial = [[1, 2], [2, 1], [0, 0]], [0.3, 1], [0.2, -0.1]
        history = true_online_td(xs, rs, lam=0, initial=initial)
        w = initial[:]
        for t in range(2):
            w, _ = semi_gradient_td(w, xs[t], rs[t], xs[t + 1], 0.9, 0.1)
            self.assertVector(history[t + 1], w)

    def test_true_online_forward_every_prefix(self):
        xs, rs = [[1, 0.2], [0.3, 1], [1, 1], [0, 0]], [0.2, -0.4, 1]
        for lam in (0, 0.3, 0.9, 1):
            for alpha in (0.01, 0.2, 0.7):
                args = dict(alpha=alpha, gamma=0.8, lam=lam, initial=[0.2, -0.3])
                for a, b in zip(true_online_td(xs, rs, **args), online_forward_view(xs, rs, **args)):
                    self.assertVector(a, b)

    def test_repeated_state_dutch_correction(self):
        self.assertAlmostEqual(traces_demo()["true_online_final"], 0.75)

    def test_zero_stepsize(self):
        history = true_online_td([[1], [1], [0]], [0, 1], alpha=0, initial=[0.3])
        self.assertEqual(history, [[0.3], [0.3], [0.3]])

    def test_empty_prefix(self):
        self.assertEqual(true_online_td([[1]], []), [[0]])

    def test_policy_gradient_finite_difference(self):
        result = policy_demo()
        self.assertAlmostEqual(result["enumerated_gradient"], result["finite_difference"], places=10)

    def test_policy_gradient_analytic(self):
        result = policy_demo()
        self.assertAlmostEqual(result["enumerated_gradient"], result["analytic_gradient"], places=12)

    def test_baseline_zero_expected_score(self):
        self.assertAlmostEqual(exact_policy_gradient(0.7, baseline=3), exact_policy_gradient(0.7), places=12)

    def test_discount_zero_ignores_later_reward(self):
        self.assertEqual(exact_policy_gradient(0.7, gamma=0), 0)

    def test_score_zero_mean(self):
        p = sigmoid(0.7)
        self.assertAlmostEqual(p * (1 - p) + (1 - p) * (-p), 0)

    def test_sigmoid_numerical_limits(self):
        self.assertEqual(sigmoid(1000), 1)
        self.assertEqual(sigmoid(-1000), 0)

    def test_gaussian_mean_score(self):
        action, mean, log_std, eps = 0.8, 0.2, -0.3, 1e-6
        log_prob = lambda mu: -log_std - 0.5 * (action - mu) ** 2 * math.exp(-2 * log_std)
        numerical = (log_prob(mean + eps) - log_prob(mean - eps)) / (2 * eps)
        self.assertAlmostEqual(gaussian_scores(action, mean, log_std)[0], numerical)

    def test_gaussian_scale_score(self):
        action, mean, log_std, eps = 0.8, 0.2, -0.3, 1e-6
        log_prob = lambda ls: -ls - 0.5 * (action - mean) ** 2 * math.exp(-2 * ls)
        numerical = (log_prob(log_std + eps) - log_prob(log_std - eps)) / (2 * eps)
        self.assertAlmostEqual(gaussian_scores(action, mean, log_std)[1], numerical)


DEMOS = {"prediction": prediction_demo, "features-control": features_demo,
         "average-control": average_demo, "off-policy": off_policy_demo,
         "traces": traces_demo, "policy-gradient": policy_demo}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["test", "demo", *DEMOS])
    args = parser.parse_args()
    if args.mode == "test":
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ApproximationTests))
        raise SystemExit(0 if result.wasSuccessful() else 1)
    result = {name: fn() for name, fn in DEMOS.items()} if args.mode == "demo" else DEMOS[args.mode]()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
