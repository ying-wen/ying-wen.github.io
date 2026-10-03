#!/usr/bin/env python3
"""Independent teaching checks, not an upstream implementation or paper reproduction.

Run: python3 formula_checks.py
Python standard library only. No downloads, file writes or training runs.
SPDX-License-Identifier: MIT
Copyright (c) 2026 Ying Wen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import math
import unittest


def gvf_return(cumulants, continuations, bootstrap=0.0):
    if len(cumulants) != len(continuations):
        raise ValueError("One continuation coefficient per transition is required")
    value = bootstrap
    for c, gamma in reversed(list(zip(cumulants, continuations))):
        if not 0 <= gamma <= 1:
            raise ValueError("Continuation must be in [0, 1]")
        value = c + gamma * value
    return value


def option_target(rewards, gamma, terminal_value):
    if not rewards:
        raise ValueError("An option must execute at least one primitive step")
    return gvf_return(rewards, [gamma] * len(rewards), terminal_value)


def average_option_step(q, rate, length, reward_sum, duration, next_q,
                        alpha=0.1, eta=0.2, beta=0.1):
    """Teaching scalar form of Wan et al. 2021 options Eq. (6)-(9).

    length is OLD expected duration, not the sampled duration.
    Returned values are simultaneous next iterates.
    """
    if length <= 0 or duration < 1:
        raise ValueError("Expected duration must be positive; sample duration >= 1")
    delta = reward_sum - rate * length + next_q - q
    return (q + alpha * delta / length,
            rate + eta * alpha * delta / length,
            length + beta * (duration - length))


class FormulaChecks(unittest.TestCase):
    def test_gvf_event_time_index(self):
        self.assertAlmostEqual(gvf_return([0, 1], [0.9, 0]), 0.9)
        self.assertAlmostEqual(option_target([0, 0], 0.9, 1), 0.81)

    def test_state_dependent_continuation(self):
        self.assertAlmostEqual(gvf_return([1, 2, 3], [0.5, 0.2, 0]), 2.3)
        self.assertEqual(gvf_return([1, 100], [0, 0.9]), 1)

    def test_rlpark_terminal_signal(self):
        r, gamma, z, next_value = 2, 0.25, 4, 8
        c = r + (1 - gamma) * z
        self.assertEqual(c + gamma * next_value, 7)

    def test_importance_sampling_expectation(self):
        mu, pi, errors = [0.25, 0.75], [1.0, 0.0], [3.0, -2.0]
        corrected = sum(b * (p / b) * d for b, p, d in zip(mu, pi, errors))
        self.assertEqual(corrected, 3)
        self.assertNotEqual(sum(b * d for b, d in zip(mu, errors)), corrected)

    def test_gtd_uses_two_different_discounts_and_old_auxiliary(self):
        # Scalar specialization of the pinned RLPark update, not copied Java.
        v, w, old_trace, x, xp = 2.0, 0.5, 0.4, 1.0, 2.0
        gamma_t, gamma_next, lam, rho = 0.5, 0.9, 0.8, 2.0
        trace = rho * (gamma_t * lam * old_trace + x)
        delta = 1 + gamma_next * v * xp - v
        correction = gamma_next * (1 - lam) * (trace * w) * xp
        vn = v + 0.1 * (delta * trace - correction)
        wn = w + 0.2 * (delta * trace - w * x * x)
        self.assertAlmostEqual(trace, 2.32)
        self.assertAlmostEqual(vn, 2.56144)
        self.assertAlmostEqual(wn, 1.6064)
        self.assertNotAlmostEqual(trace, rho * (gamma_next * lam * old_trace + x))
        self.assertNotAlmostEqual(vn, v + 0.1 * (delta * trace - gamma_next * (1-lam) * trace * wn * xp))

    def test_option_termination_mixture(self):
        continuation = (1 - 0.25) * 2 + 0.25 * 6
        self.assertEqual(continuation, 3)
        self.assertAlmostEqual(1 + 0.9 * continuation, 3.7)
        self.assertEqual(1 + 0.9 * (1 - 1) * continuation, 1)

    def test_termination_gradient_finite_difference(self):
        sigmoid = lambda x: 1 / (1 + math.exp(-x))
        theta, advantage, eps = 0.3, -4.0, 1e-6
        objective = lambda x: -sigmoid(x) * advantage
        numeric = (objective(theta + eps) - objective(theta - eps)) / (2 * eps)
        analytic = -sigmoid(theta) * (1 - sigmoid(theta)) * advantage
        self.assertAlmostEqual(numeric, analytic, places=8)
        self.assertGreater(sigmoid(theta + 0.1 * analytic), sigmoid(theta))

    def test_smdp_discount_power(self):
        self.assertAlmostEqual(option_target([-1, -1, 0], 0.9, 10), 5.39)
        self.assertAlmostEqual(option_target([2], 0.9, 10), 11)

    def test_discounted_option_model_no_extra_gamma(self):
        gamma = 0.9
        r_model = -1 - gamma
        p_model = gamma ** 3
        self.assertAlmostEqual(r_model + p_model * 10, 5.39)
        self.assertNotAlmostEqual(r_model + gamma * p_model * 10, 5.39)

    def test_random_duration_is_inside_expectation(self):
        exact = 0.5 * (0.9 + 0.9 ** 3) * 10
        self.assertAlmostEqual(exact, 8.145)
        self.assertNotAlmostEqual(exact, 0.9 ** 2 * 10)

    def test_renewal_reward_rate_not_mean_of_ratios(self):
        rate = (0.5 * 1 + 0.5 * 1) / (0.5 * 1 + 0.5 * 3)
        wrong = 0.5 * (1 / 1) + 0.5 * (1 / 3)
        self.assertEqual(rate, 0.5)
        self.assertAlmostEqual(wrong, 2 / 3)
        # Mean residual of sample-normalized updates is nonzero at the true rate.
        self.assertNotAlmostEqual(0.5 * (1 - rate) + 0.5 * (1 - 3 * rate) / 3, 0)

    def test_average_option_simultaneous_old_length(self):
        q, rate, length = average_option_step(2, 0.5, 2, 3, 4, 5)
        self.assertAlmostEqual(q, 2.25)
        self.assertAlmostEqual(rate, 0.55)
        self.assertAlmostEqual(length, 2.2)
        with self.assertRaises(ValueError):
            average_option_step(2, 0.5, 0, 3, 4, 5)

    def test_constant_reward_centering_value_shift(self):
        reward, baseline, gamma = 3, 1, 0.9
        uncentered = reward / (1 - gamma)
        centered = (reward - baseline) / (1 - gamma)
        self.assertAlmostEqual(uncentered - centered, baseline / (1 - gamma))

    def test_rvi_sac_target_keeps_continuation_after_reset(self):
        reward, cost, reset, fq, soft_next = 2, 3, 1, 0.5, 4
        target = reward - cost * reset - fq + soft_next
        self.assertEqual(target, 2.5)
        self.assertNotEqual(target, reward - cost * reset - fq + (1-reset)*soft_next)

    def test_expectation_model_requires_linearity(self):
        xs, probs, w = [-1, 1], [0.5, 0.5], 3
        mean_x = sum(p*x for p, x in zip(probs, xs))
        self.assertEqual(sum(p*w*x for p, x in zip(probs, xs)), w*mean_x)
        self.assertNotEqual(sum(p*x*x for p, x in zip(probs, xs)), mean_x**2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
