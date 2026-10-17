#!/usr/bin/env python3
"""State construction and meta-learning mechanism laboratory (MIT).

Python 3.10+, standard library only. These are independently written teaching
implementations, not reproductions of RTU, PEARL, or other paper benchmarks.
Use: python3 state_meta_lab.py all | state | meta | test
"""
from __future__ import annotations

import argparse
import math
import random
import unittest


# BEGIN belief
def belief_step(prior, transition, likelihood):
    """P[i][j] = P(next=j | current=i, selected_action)."""
    n = len(prior)
    if len(transition) != n or len(likelihood) != n:
        raise ValueError("inconsistent state dimensions")
    if any(len(row) != n for row in transition):
        raise ValueError("transition must be square")
    if any(v < 0 for v in prior + likelihood):
        raise ValueError("negative probability")
    if not math.isclose(sum(prior), 1.0):
        raise ValueError("prior must sum to one")
    if any(any(v < 0 for v in row) or not math.isclose(sum(row), 1.0)
           for row in transition):
        raise ValueError("each transition row must be a probability distribution")
    predicted = [sum(prior[i] * transition[i][j] for i in range(n))
                 for j in range(n)]
    unnormalized = [predicted[j] * likelihood[j] for j in range(n)]
    evidence = sum(unnormalized)
    if evidence <= 0:
        raise ValueError("observation has zero probability under this model")
    return [v / evidence for v in unnormalized]
# END belief


# BEGIN rtrl
def rtrl(theta, inputs, initial=0.0):
    """Exact derivatives for one FIXED parameter vector and fixed initial state.

    h[t] = tanh(a*h[t-1] + b*x[t] + c), theta=(a,b,c).
    Parameters are not changed while processing the sequence.
    """
    a, b, c = theta
    h, eligibility = initial, [0.0, 0.0, 0.0]
    states, derivatives = [], []
    for x in inputs:
        old_h = h
        h = math.tanh(a * old_h + b * x + c)
        local = [old_h, x, 1.0]
        eligibility = [(1.0 - h * h) * (a * e + d)
                       for e, d in zip(eligibility, local)]
        states.append(h)
        derivatives.append(eligibility[:])
    return states, derivatives


def final_loss_grad(theta, inputs, target, initial=0.0):
    states, jacobians = rtrl(theta, inputs, initial)
    error = states[-1] - target
    return 0.5 * error * error, [error * e for e in jacobians[-1]]
# END rtrl


# BEGIN bptt
def bptt_final(theta, inputs, target, window=None, initial=0.0):
    """Final-state loss; window=None is full BPTT.

    A finite window holds the earlier boundary state numerically intact but
    treats it as a constant during differentiation (detach, not state reset).
    """
    states, _ = rtrl(theta, inputs, initial)
    a = theta[0]
    left = 0 if window is None else max(0, len(inputs) - window)
    adjoint = states[-1] - target
    grad = [0.0, 0.0, 0.0]
    for t in range(len(inputs) - 1, left - 1, -1):
        previous = initial if t == 0 else states[t - 1]
        dz = adjoint * (1.0 - states[t] ** 2)
        for j, feature in enumerate([previous, inputs[t], 1.0]):
            grad[j] += dz * feature
        adjoint = a * dz
    return grad
# END bptt


# BEGIN rtu
def rotation_trace(inputs, radius=0.9, angle=0.4, input_weights=(0.7, -0.2)):
    """One linear RTU-style rotation block and exact r/angle sensitivities.

    Radius is differentiated directly; a paper implementation usually learns
    log-log radius parameters instead. No actor, critic, optimizer, or benchmark
    is included. Both new components use OLD recurrent components.
    """
    if not 0.0 < radius < 1.0:
        raise ValueError("radius must be strictly inside (0,1)")
    cs, sn = math.cos(angle), math.sin(angle)
    norm = math.sqrt(1.0 - radius * radius)
    h, dr, dp = [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]

    def rotate(v):
        return [cs * v[0] - sn * v[1], sn * v[0] + cs * v[1]]

    for x in inputs:
        rotated, er, ep = rotate(h), rotate(dr), rotate(dp)
        angle_direct = [-sn * h[0] - cs * h[1],
                        cs * h[0] - sn * h[1]]
        dr = [radius * er[j] + rotated[j]
              - radius / norm * input_weights[j] * x for j in range(2)]
        dp = [radius * ep[j] + radius * angle_direct[j] for j in range(2)]
        h = [radius * rotated[j] + norm * input_weights[j] * x for j in range(2)]
    return h, dr, dp
# END rtu


# BEGIN state_experiment
def delayed_cue(epochs=1400, seed=3):
    """Controlled, resettable SUPERVISED sequence experiment, not lifelong RL."""
    rng = random.Random(seed)
    theta = [0.8, 0.3, 0.0]
    for _ in range(epochs):
        cue = rng.choice([-1.0, 1.0])
        inputs, target = [cue, 0.0, 0.0, 0.0], 0.8 * cue
        _, grad = final_loss_grad(theta, inputs, target)
        # Sequence starts from h=0; weights change only AFTER its full gradient.
        theta = [w - 0.08 * g for w, g in zip(theta, grad)]
    predictions = [rtrl(theta, [cue, 0.0, 0.0, 0.0])[0][-1]
                   for cue in [-1.0, 1.0]]
    return theta, predictions


def state_demo():
    prior = [0.5, 0.5]
    identity = [[1.0, 0.0], [0.0, 1.0]]
    first = belief_step(prior, identity, [0.8, 0.2])
    second = belief_step(first, identity, [0.8, 0.2])
    print("belief, one/two red cues:", first, second)
    theta, inputs, target = [0.8, 0.3, 0.0], [1.0, 0.0, 0.0, 0.0], 0.8
    loss, forward = final_loss_grad(theta, inputs, target)
    backward = bptt_final(theta, inputs, target)
    truncated = bptt_final(theta, inputs, target, window=2)
    print("RTRL gradient:", forward)
    print("BPTT gradient:", backward)
    print("TBPTT-2 gradient:", truncated, "(cue input-weight gradient is zero)")
    learned, predictions = delayed_cue()
    print("trained cue predictions:", predictions)
    print("rotation block state, d/dr, d/dangle:", rotation_trace([1, 0, 0]))
    print("scope: known-model filtering + fixed-parameter derivatives + supervised sequences")
# END state_experiment


# BEGIN sensitivity
def fixed_beta_training(beta, train_targets, validation_target, initial=0.0):
    """Exact unrolled log-step-size derivative for FIXED beta and fixed data.

    w_next = w + exp(beta)*(target-w). H=dw/dbeta, H0=0.
    Includes every inner update Jacobian; it is not an online changing-beta
    total derivative and does not differentiate the data collection policy.
    """
    alpha = math.exp(beta)
    w, h = initial, 0.0
    trace = []
    for target in train_targets:
        error = target - w
        h = (1.0 - alpha) * h + alpha * error  # use old w and H
        w = w + alpha * error
        trace.append((w, h))
    loss = 0.5 * (w - validation_target) ** 2
    hypergradient = (w - validation_target) * h
    return loss, hypergradient, trace
# END sensitivity


# BEGIN idbd
def idbd_step(weights, beta, history, features, target, meta_rate=0.01):
    """Linear supervised IDBD, with diagonal sensitivity and positive clipping."""
    error = target - sum(w * x for w, x in zip(weights, features))
    new_beta = [b + meta_rate * error * x * h
                for b, x, h in zip(beta, features, history)]
    alpha = [math.exp(b) for b in new_beta]
    new_weights = [w + a * error * x
                   for w, a, x in zip(weights, alpha, features)]
    new_history = [h * max(0.0, 1.0 - a * x * x) + a * error * x
                   for h, a, x in zip(history, alpha, features)]
    return new_weights, new_beta, new_history, error
# END idbd


# BEGIN tidbd
def tidbd_step(weights, beta, history, eligibility, features, next_features,
               reward, gamma=0.9, lam=0.8, meta_rate=0.01):
    """2018 TIDBD(lambda) semi-gradient version, on-policy, constant gamma.

    This is NOT the later full TD-error-gradient/normalization variants.
    When the protocol declares a genuine new episode, the CALLER clears z.
    """
    value = sum(w * x for w, x in zip(weights, features))
    next_value = sum(w * x for w, x in zip(weights, next_features))
    delta = reward + gamma * next_value - value
    new_beta = [b + meta_rate * delta * x * h
                for b, x, h in zip(beta, features, history)]
    alpha = [math.exp(b) for b in new_beta]
    new_z = [gamma * lam * z + x for z, x in zip(eligibility, features)]
    new_weights = [w + a * delta * z
                   for w, a, z in zip(weights, alpha, new_z)]
    new_history = [h * max(0.0, 1.0 - a * x * z) + a * delta * z
                   for h, a, x, z in zip(history, alpha, features, new_z)]
    return new_weights, new_beta, new_history, new_z, delta
# END tidbd


# BEGIN maml
def scalar_maml(initialization, train_target, test_target, curvature=2.0, alpha=0.1):
    """One task, deterministic quadratic losses, exact MAML and FO approximation.

    L_train = curvature/2*(w-train_target)^2; L_test = (w-test_target)^2/2.
    This illustrates optimizer differentiation, not policy-gradient sampling.
    """
    adapted = initialization - alpha * curvature * (initialization - train_target)
    outer_gradient = adapted - test_target
    exact = (1.0 - alpha * curvature) * outer_gradient
    first_order = outer_gradient
    return 0.5 * outer_gradient ** 2, exact, first_order, adapted
# END maml


# BEGIN sampling
def stochastic_adaptation(theta, alpha=0.2, target=1.0):
    """Enumerate a Bernoulli sample's TWO gradient paths exactly.

    a~Bernoulli(sigmoid(theta)); w=(1-alpha)*theta+alpha*a.
    The toy outer loss is (w-target)^2/2. No Monte Carlo noise is present.
    """
    probability = 1.0 / (1.0 + math.exp(-theta))
    expected_loss = path_gradient = score_gradient = 0.0
    for action, mass in [(0.0, 1.0 - probability), (1.0, probability)]:
        adapted = (1.0 - alpha) * theta + alpha * action
        loss = 0.5 * (adapted - target) ** 2
        expected_loss += mass * loss
        path_gradient += mass * (adapted - target) * (1.0 - alpha)
        score_gradient += mass * loss * (action - probability)
    return expected_loss, path_gradient, score_gradient
# END sampling


# BEGIN return_sensitivity
def lambda_return_sensitivity(rewards, next_values, bootstrap, gamma=0.9, lam=0.8):
    """Lambda return and exact direct derivatives for fixed values/boundary."""
    if len(rewards) != len(next_values):
        raise ValueError("one next-state value is required per reward")
    g, dg, dl = bootstrap, 0.0, 0.0
    for reward, value in reversed(list(zip(rewards, next_values))):
        mixture = (1.0 - lam) * value + lam * g
        new_dg = mixture + gamma * lam * dg
        new_dl = gamma * (g - value + lam * dl)
        g = reward + gamma * mixture
        dg, dl = new_dg, new_dl
    return g, dg, dl
# END return_sensitivity


# BEGIN context
def gaussian_context(observations, noise_variance=1.0, prior_mean=0.0,
                     prior_variance=1.0):
    """Exact Bayesian latent-task inference for y_i=z+Gaussian noise.

    It demonstrates context-driven adaptation with NO parameter gradient.
    It is not PEARL's learned encoder or SAC training loop.
    """
    if noise_variance <= 0 or prior_variance <= 0:
        raise ValueError("variances must be positive")
    precision = 1.0 / prior_variance + len(observations) / noise_variance
    natural_mean = prior_mean / prior_variance + sum(observations) / noise_variance
    return natural_mean / precision, 1.0 / precision
# END context


# BEGIN meta_experiment
def meta_demo():
    loss, grad, trace = fixed_beta_training(math.log(0.1), [2.0, 2.0], 3.0)
    print("two inner steps (w,H):", trace)
    print("validation loss, d/dbeta:", loss, grad)
    w, b, h = [0.0], [math.log(0.1)], [0.0]
    for target in [2.0, 2.0, -2.0]:
        w, b, h, delta = idbd_step(w, b, h, [1.0], target)
        print("IDBD: target,w,alpha,H:", target, w[0], math.exp(b[0]), h[0])
    result = tidbd_step([0.0, 0.0], [math.log(0.1)] * 2, [0.0] * 2,
                       [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], 1.0)
    print("TIDBD: weights, traces:", result[0], result[3])
    print("MAML: loss, exact, FO, adapted:", scalar_maml(0.0, 2.0, 3.0))
    print("stochastic adaptation loss,path,score:", stochastic_adaptation(0.3))
    print("lambda return, d/dgamma, d/dlambda:",
          lambda_return_sensitivity([1, 2], [0.3, 0.4], 0.4))
    for context in [[], [1.0], [1.0, 1.0, 1.0]]:
        print("context posterior (mean,var):", context, gaussian_context(context))
    print("scope: analytic mechanism tests, not meta-RL benchmark performance")
# END meta_experiment


def central_difference(fn, value, eps=1e-6):
    return (fn(value + eps) - fn(value - eps)) / (2.0 * eps)


class StateMetaTests(unittest.TestCase):
    def test_belief_two_observations(self):
        p = belief_step([0.5, 0.5], [[1, 0], [0, 1]], [0.8, 0.2])
        p = belief_step(p, [[1, 0], [0, 1]], [0.8, 0.2])
        self.assertAlmostEqual(p[0], 16 / 17)

    def test_belief_dynamics_before_observation(self):
        p = belief_step([1.0, 0.0], [[0.9, 0.1], [0.2, 0.8]], [0.2, 0.8])
        self.assertAlmostEqual(p[0], 0.18 / 0.26)

    def test_belief_impossible_observation(self):
        with self.assertRaises(ValueError):
            belief_step([0.5, 0.5], [[1, 0], [0, 1]], [0, 0])

    def test_belief_invalid_transition(self):
        with self.assertRaises(ValueError):
            belief_step([0.5, 0.5], [[1, 1], [0, 1]], [0.8, 0.2])

    def test_rtrl_finite_difference_all_parameters(self):
        theta, xs, y = [0.8, 0.3, -0.05], [1.0, 0.0, -0.2, 0.0], 0.8
        _, gradient = final_loss_grad(theta, xs, y)
        for j in range(3):
            def fn(v):
                copied = theta[:]
                copied[j] = v
                return final_loss_grad(copied, xs, y)[0]
            self.assertAlmostEqual(gradient[j], central_difference(fn, theta[j]), places=7)

    def test_rtrl_equals_bptt(self):
        theta, xs, y = [0.8, 0.3, -0.05], [1, 0, -0.2, 0], 0.8
        _, fg = final_loss_grad(theta, xs, y)
        bg = bptt_final(theta, xs, y)
        for a, b in zip(fg, bg):
            self.assertAlmostEqual(a, b, places=12)

    def test_full_window_equals_full_bptt(self):
        args = ([0.8, 0.3, 0], [1, 0, 0, 0], 0.8)
        self.assertEqual(bptt_final(*args), bptt_final(*args, window=4))

    def test_tbptt_loses_cue_gradient_not_state(self):
        theta, xs, y = [0.8, 0.3, 0.0], [1, 0, 0, 0], 0.8
        self.assertEqual(bptt_final(theta, xs, y, window=2)[1], 0.0)
        self.assertNotAlmostEqual(bptt_final(theta, xs, y)[1], 0.0)
        self.assertGreater(rtrl(theta, xs)[0][-1], 0.0)

    def test_delayed_cue_training(self):
        _, predictions = delayed_cue()
        self.assertLess(abs(predictions[0] + 0.8), 0.03)
        self.assertLess(abs(predictions[1] - 0.8), 0.03)

    def test_rotation_radius_derivative(self):
        xs = [1.0, -0.3, 0.0, 0.1]
        _, dr, _ = rotation_trace(xs)
        for j in range(2):
            fd = central_difference(lambda r: rotation_trace(xs, radius=r)[0][j], 0.9)
            self.assertAlmostEqual(dr[j], fd, places=7)

    def test_rotation_angle_derivative(self):
        xs = [1.0, -0.3, 0.0, 0.1]
        _, _, dp = rotation_trace(xs)
        for j in range(2):
            fd = central_difference(lambda p: rotation_trace(xs, angle=p)[0][j], 0.4)
            self.assertAlmostEqual(dp[j], fd, places=7)

    def test_rotation_zero_input_no_signal(self):
        h, dr, dp = rotation_trace([0, 0, 0])
        self.assertEqual(h + dr + dp, [0.0] * 6)

    def test_fixed_beta_hypergradient(self):
        b = math.log(0.1)
        loss, gradient, trace = fixed_beta_training(b, [2.0, -0.5, 1.2], 3.0)
        fd = central_difference(lambda x: fixed_beta_training(x, [2, -0.5, 1.2], 3)[0], b)
        self.assertAlmostEqual(gradient, fd, places=7)

    def test_fixed_beta_hand_calculation(self):
        loss, gradient, trace = fixed_beta_training(math.log(0.1), [2, 2], 3)
        self.assertAlmostEqual(trace[-1][0], 0.38)
        self.assertAlmostEqual(trace[-1][1], 0.36)
        self.assertAlmostEqual(loss, 3.4322)
        self.assertAlmostEqual(gradient, -0.9432)

    def test_idbd_first_step_no_meta_signal(self):
        w, b, h, _ = idbd_step([0], [math.log(0.1)], [0], [1], 2)
        self.assertAlmostEqual(w[0], 0.2)
        self.assertAlmostEqual(h[0], 0.2)
        self.assertEqual(b[0], math.log(0.1))

    def test_idbd_consistent_updates_raise_alpha(self):
        w, b, h, _ = idbd_step([0], [math.log(0.1)], [0], [1], 2)
        _, new_b, _, _ = idbd_step(w, b, h, [1], 2)
        self.assertGreater(new_b[0], b[0])

    def test_tidbd_reduces_to_idbd_gamma_zero(self):
        args = ([0.2], [math.log(0.1)], [0.4])
        iw, ib, ih, ie = idbd_step(*args, [1.0], 2.0)
        tw, tb, th, tz, td = tidbd_step(*args, [0.0], [1.0], [0.3], 2.0, gamma=0)
        for a, b in zip(iw + ib + ih + [ie], tw + tb + th + [td]):
            self.assertAlmostEqual(a, b)

    def test_tidbd_trace_carries_credit(self):
        out = tidbd_step([0, 0], [math.log(0.1)] * 2, [0, 0], [1, 0],
                         [0, 1], [1, 0], 1)
        self.assertAlmostEqual(out[0][0], 0.072)
        self.assertAlmostEqual(out[0][1], 0.1)

    def test_maml_exact_finite_difference(self):
        _, exact, fo, _ = scalar_maml(0.0, 2.0, 3.0)
        fd = central_difference(lambda w: scalar_maml(w, 2, 3)[0], 0.0)
        self.assertAlmostEqual(exact, fd, places=7)
        self.assertAlmostEqual(exact, -2.08)
        self.assertAlmostEqual(fo, -2.6)

    def test_maml_fo_only_equal_zero_curvature(self):
        _, exact, fo, _ = scalar_maml(0, 2, 3, curvature=0)
        self.assertEqual(exact, fo)

    def test_sampling_path_plus_score_finite_difference(self):
        _, path, score = stochastic_adaptation(0.3)
        fd = central_difference(lambda theta: stochastic_adaptation(theta)[0], 0.3)
        self.assertAlmostEqual(path + score, fd, places=7)
        self.assertNotAlmostEqual(path, fd, places=4)

    def test_return_gamma_finite_difference(self):
        _, dg, _ = lambda_return_sensitivity([1, 2], [0.3, 0.4], 0.4)
        fd = central_difference(lambda g: lambda_return_sensitivity(
            [1, 2], [0.3, 0.4], 0.4, gamma=g)[0], 0.9)
        self.assertAlmostEqual(dg, fd, places=7)

    def test_return_lambda_finite_difference(self):
        _, _, dl = lambda_return_sensitivity([1, 2], [0.3, 0.4], 0.4)
        fd = central_difference(lambda l: lambda_return_sensitivity(
            [1, 2], [0.3, 0.4], 0.4, lam=l)[0], 0.8)
        self.assertAlmostEqual(dl, fd, places=7)

    def test_return_lambda_endpoints(self):
        zero, _, _ = lambda_return_sensitivity([1, 2], [0.3, 0.4], 0.4, lam=0)
        one, _, _ = lambda_return_sensitivity([1, 2], [0.3, 0.4], 0.4, lam=1)
        self.assertAlmostEqual(zero, 1 + 0.9 * 0.3)
        self.assertAlmostEqual(one, 1 + 0.9 * 2 + 0.9 ** 2 * 0.4)

    def test_context_prior(self):
        self.assertEqual(gaussian_context([]), (0.0, 1.0))

    def test_context_permutation_invariance(self):
        self.assertEqual(gaussian_context([1, 2, 3]), gaussian_context([3, 1, 2]))

    def test_context_information_and_prediction(self):
        m1, v1 = gaussian_context([1])
        m3, v3 = gaussian_context([1, 1, 1])
        self.assertEqual((m1, v1), (0.5, 0.5))
        self.assertEqual((m3, v3), (0.75, 0.25))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", nargs="?", default="all",
                        choices=["all", "state", "meta", "test"])
    args = parser.parse_args()
    if args.experiment == "test":
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(StateMetaTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    if args.experiment in ("all", "state"):
        state_demo()
    if args.experiment in ("all", "meta"):
        meta_demo()


if __name__ == "__main__":
    main()
