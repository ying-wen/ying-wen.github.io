"""Standard-library numerical kernels for the six deep-RL lessons.

python3 deep_textbook_lab.py demo
python3 deep_textbook_lab.py test
These kernels are not complete Atari/MuJoCo training implementations.
"""
import argparse
import json
import math
import unittest


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def softplus(x):
    return max(x, 0.0) + math.log1p(math.exp(-abs(x)))


# BEGIN targets
def dqn_target(reward, gamma, terminated, next_online, next_target,
               double=True):
    if not next_online or len(next_online) != len(next_target):
        raise ValueError('matching nonempty action vectors required')
    if terminated:
        return reward
    if double:
        selected = max(range(len(next_online)), key=next_online.__getitem__)
        continuation = next_target[selected]
    else:
        continuation = max(next_target)
    return reward + gamma * continuation


def masks(terminated, boundary):
    # boundary includes terminal, timeout, or the end of this rollout.
    return float(not terminated), float(not (terminated or boundary))
# END targets


# BEGIN advantages
def gae(rewards, values, next_values, terminated, boundaries,
        gamma=0.99, lam=0.95):
    n = len(rewards)
    if not all(len(x) == n for x in
               (values, next_values, terminated, boundaries)):
        raise ValueError('one next value and two masks per transition')
    advantages, carry = [0.0] * n, 0.0
    for t in reversed(range(n)):
        bootstrap, trace = masks(terminated[t], boundaries[t])
        delta = rewards[t] + gamma * bootstrap * next_values[t] - values[t]
        carry = delta + gamma * lam * trace * carry
        advantages[t] = carry
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


def score_gradient(probabilities, action, advantage):
    # Derivative of A * log softmax(logits)[action] with fixed A.
    return [advantage * (float(i == action) - p)
            for i, p in enumerate(probabilities)]
# END advantages


# BEGIN trust_region
def ppo_term(ratio, advantage, clip=0.2):
    clipped = min(1 + clip, max(1 - clip, ratio))
    return min(ratio * advantage, clipped * advantage)


def conjugate_gradient(matvec, b, iterations=20, tolerance=1e-12):
    x, residual = [0.0] * len(b), list(b)
    direction = list(residual)
    rr = dot(residual, residual)
    for _ in range(iterations):
        if rr <= tolerance * tolerance:
            break
        product = matvec(direction)
        curvature = dot(direction, product)
        if curvature <= 0:
            raise ValueError('positive curvature is required')
        step = rr / curvature
        x = [v + step * d for v, d in zip(x, direction)]
        residual = [r - step * p for r, p in zip(residual, product)]
        new_rr = dot(residual, residual)
        direction = [r + new_rr / rr * d
                     for r, d in zip(residual, direction)]
        rr = new_rr
    return x


def categorical_kl(old, new):
    return sum(p * math.log(p / q) for p, q in zip(old, new) if p > 0)


def trpo_binary(theta, advantage_one, advantage_zero, delta=0.01):
    """Exact scalar Fisher + actual KL/line search for one-state policy."""
    prob = lambda z: 1.0 / (1.0 + math.exp(-z))
    old_p = prob(theta)
    surrogate = lambda z: prob(z) * advantage_one + (1-prob(z)) * advantage_zero
    gradient = old_p * (1-old_p) * (advantage_one-advantage_zero)
    if abs(gradient) < 1e-14:
        return theta, 0.0, 0.0
    fisher = old_p * (1-old_p)
    natural = gradient / fisher
    full_step = math.sqrt(2*delta/(natural*fisher*natural)) * natural
    for power in range(20):
        step = 0.5**power * full_step
        candidate = theta + step
        kl = categorical_kl([old_p, 1-old_p], [prob(candidate), 1-prob(candidate)])
        gain = surrogate(candidate)-surrogate(theta)
        if kl <= delta and gain >= 0.1 * gradient * step:
            return candidate, kl, gain
    return theta, 0.0, 0.0
# END trust_region


# BEGIN continuous
def td3_target(reward, gamma, terminated, action, noise, noise_clip,
               low, high, q1, q2):
    perturbation = max(-noise_clip, min(noise_clip, noise))
    smoothed = max(low, min(high, action + perturbation))
    target = reward + gamma * (not terminated) * min(q1(smoothed), q2(smoothed))
    return target, smoothed


def polyak(old, online, tau):
    if not 0 <= tau <= 1:
        raise ValueError('tau must lie in [0, 1]')
    return [(1-tau)*x + tau*y for x, y in zip(old, online)]


def tanh_log_prob(u, mean, log_std, scale=1.0):
    if scale <= 0:
        raise ValueError('positive affine action scale required')
    log_normal = -0.5*((u-mean)/math.exp(log_std))**2-log_std-0.5*math.log(2*math.pi)
    log_jacobian = 2*(math.log(2)-u-softplus(-2*u))
    return log_normal - log_jacobian - math.log(scale)


def sac_target(reward, gamma, terminated, q1, q2, log_prob, alpha):
    return reward + gamma*(not terminated)*(min(q1, q2)-alpha*log_prob)


def temperature_log_gradient(log_alpha, log_prob, target_entropy):
    # Exact gradient of -exp(log_alpha) * stopgrad(log_prob + target_entropy).
    return -math.exp(log_alpha)*(log_prob+target_entropy)
# END continuous


# BEGIN interfaces
def recurrent_derivative(a, b, inputs):
    state, sensitivity = 0.0, 0.0
    for observation in inputs:
        state = a*state + b*observation
        sensitivity = a*sensitivity + observation
    return state, sensitivity


def conservative_penalty(q_values, data_action):
    maximum = max(q_values)
    return maximum + math.log(sum(math.exp(q-maximum) for q in q_values))-q_values[data_action]


def bootstrap_bias_bound(one_step_error, gamma):
    if one_step_error < 0 or not 0 <= gamma < 1:
        raise ValueError('nonnegative error and discount below one required')
    return one_step_error/(1-gamma)
# END interfaces


class Checks(unittest.TestCase):
    def test_double_selection_evaluation(self):
        self.assertAlmostEqual(dqn_target(1, .9, False, [5, 4], [2, 6]), 2.8)
        self.assertAlmostEqual(dqn_target(1, .9, False, [5, 4], [2, 6], False), 6.4)

    def test_terminal_target(self):
        self.assertEqual(dqn_target(3, .9, True, [999], [999]), 3)

    def test_timeout_two_masks(self):
        self.assertEqual(masks(False, True), (1, 0))
        self.assertEqual(masks(True, False), (0, 0))

    def test_gae_manual(self):
        a, ret = gae([1, 2], [.5, 1], [1, 0], [False, True], [False, True], .9, .8)
        self.assertAlmostEqual(a[0], 2.12)
        self.assertAlmostEqual(a[1], 1)
        self.assertAlmostEqual(ret[0], 2.62)

    def test_gae_no_cross_episode(self):
        a, _ = gae([0, 100], [1, 0], [2, 0], [False, True], [True, True], .9, 1)
        self.assertAlmostEqual(a[0], .8)

    def test_gae_lambda_zero(self):
        a, _ = gae([1, 2], [.5, 1], [1, 0], [False, True], [False, True], .9, 0)
        self.assertEqual(a, [1.4, 1.0])

    def test_score_gradient(self):
        self.assertEqual(score_gradient([.25, .75], 0, 2), [1.5, -1.5])

    def test_ppo_sign(self):
        self.assertAlmostEqual(ppo_term(1.4, 2), 2.4)
        self.assertAlmostEqual(ppo_term(.6, -2), -1.6)
        self.assertAlmostEqual(ppo_term(1.4, -2), -2.8)

    def test_conjugate_gradient(self):
        solution = conjugate_gradient(lambda x: [4*x[0]+x[1], x[0]+3*x[1]], [1, 2])
        self.assertAlmostEqual(solution[0], 1/11)
        self.assertAlmostEqual(solution[1], 7/11)

    def test_cg_zero_and_curvature(self):
        self.assertEqual(conjugate_gradient(lambda x: x, [0, 0]), [0, 0])
        with self.assertRaises(ValueError):
            conjugate_gradient(lambda x: [-v for v in x], [1])

    def test_trpo_acceptance(self):
        theta, kl, gain = trpo_binary(0, 1, -1)
        self.assertGreater(theta, 0)
        self.assertLessEqual(kl, .01)
        self.assertGreater(gain, 0)

    def test_td3_noise_clipping(self):
        target, action = td3_target(1, .9, False, .9, 3, .2, -1, 1,
                                    lambda a: 2+a, lambda a: 1+a)
        self.assertEqual(action, 1)
        self.assertAlmostEqual(target, 2.8)

    def test_polyak_convention(self):
        self.assertEqual(polyak([0, 2], [10, 4], .1), [1, 2.2])

    def test_tanh_jacobian(self):
        u = .7
        expected = -.5*u*u-.5*math.log(2*math.pi)-math.log(1-math.tanh(u)**2)
        self.assertAlmostEqual(tanh_log_prob(u, 0, 0), expected)
        self.assertTrue(math.isfinite(tanh_log_prob(50, 0, 0)))

    def test_affine_action_scale(self):
        self.assertAlmostEqual(tanh_log_prob(.5, 0, 0, 2)-tanh_log_prob(.5, 0, 0), -math.log(2))

    def test_soft_target_and_temperature(self):
        self.assertAlmostEqual(sac_target(1, .9, False, 3, 2, -.5, .2), 2.89)
        gradient = temperature_log_gradient(math.log(.2), 2, -1)
        self.assertAlmostEqual(gradient, -.2)
        eps = 1e-6
        loss = lambda x: -math.exp(x)*(2-1)
        self.assertAlmostEqual(gradient, (loss(math.log(.2)+eps)-loss(math.log(.2)-eps))/(2*eps), places=8)

    def test_recurrent_finite_difference(self):
        h, derivative = recurrent_derivative(.5, .2, [1, 0, 0])
        eps = 1e-6
        fd = (recurrent_derivative(.5, .2+eps, [1, 0, 0])[0]-recurrent_derivative(.5, .2-eps, [1, 0, 0])[0])/(2*eps)
        self.assertAlmostEqual(h, .05)
        self.assertAlmostEqual(derivative, fd)

    def test_offline_penalty(self):
        self.assertGreater(conservative_penalty([0, 10], 0), 10)
        self.assertAlmostEqual(conservative_penalty([0, 0], 0), math.log(2))

    def test_model_bound(self):
        self.assertAlmostEqual(bootstrap_bias_bound(.02, .9), .2)


def demo():
    print(json.dumps({
        'dqn_vs_double': [dqn_target(1, .9, False, [5, 4], [2, 6], False), dqn_target(1, .9, False, [5, 4], [2, 6])],
        'gae': gae([1, 2], [.5, 1], [1, 0], [False, True], [False, True], .9, .8),
        'ppo': [ppo_term(1.4, 2), ppo_term(.6, -2)],
        'trpo_binary': trpo_binary(0, 1, -1),
        'sac_target': sac_target(1, .9, False, 3, 2, -.5, .2),
        'recurrent': recurrent_derivative(.5, .2, [1, 0, 0]),
    }, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['demo', 'test'])
    args = parser.parse_args()
    if args.command == 'test':
        unittest.main(argv=['deep_textbook_lab.py'], verbosity=2)
    else:
        demo()
