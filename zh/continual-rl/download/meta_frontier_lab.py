"""Small, standard-library experiments for meta-gradients and output-space steps.

Run: python3 meta_frontier_lab.py all | metatrace | intentional | test
Independent teaching implementation, not the authors' benchmark training code.
Code: MIT. See the website chapter for derivations and primary sources.
"""
from dataclasses import dataclass, field
import argparse
import math
import unittest


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


# BEGIN metatrace
@dataclass
class LinearMetatrace:
    """Scalar, critic-only, unnormalized Metatrace core (no entropy term).

    U(w,x)=V(w,x)=w*x. For mu=0 and fixed data, h is exact dw/d(log alpha).
    With online beta updates, h is the local sensitivity approximation.
    """
    w: float = 0.0
    beta: float = math.log(0.1)
    mu: float = 0.01
    gamma: float = 0.9
    lam: float = 0.8
    z: float = 0.0
    h: float = 0.0
    z_beta: float = 0.0

    def step(self, x, reward, next_x, terminal=False):
        gamma = 0.0 if terminal else self.gamma
        delta = reward + gamma * self.w * next_x - self.w * x
        # Old h belongs to the weights that generated the current prediction.
        old_h = self.h
        self.z = self.gamma * self.lam * self.z + x
        self.z_beta = self.gamma * self.lam * self.z_beta + x * old_h
        self.beta += self.mu * delta * self.z_beta
        alpha = math.exp(self.beta)
        delta_derivative = gamma * next_x - x
        self.h = old_h + alpha * self.z * (delta + delta_derivative * old_h)
        self.w += alpha * delta * self.z
        result = dict(w=self.w, h=self.h, alpha=alpha, delta=delta,
                      z=self.z, z_beta=self.z_beta)
        if terminal:
            # Genuine episodic boundary; parameters and step-size persist.
            self.z = self.z_beta = 0.0
        return result
# END metatrace


# BEGIN intentional
@dataclass
class IntentionalStep:
    """Vector optimizer matching the full author's default normalization order.

    The caller supplies +grad V for the critic. For the author's actor use
    +grad_theta[log pi(a|s) + xi*H(pi(.|s))*sign(delta)], with the sign of the
    pre-update critic's delta held constant. Entropy enters the trace and the
    RMSProp statistics with the score gradient; it is not added after the step.
    Clipping and positive-scale normalization preserve delta's sign. Omitting
    this sign reverses the immediate entropy contribution for negative delta.
    With nonzero traces, past entropy terms may have different signs; this is
    not a guarantee that each update increases the current state's entropy.
    'policy' additionally normalizes the clipped TD error by its running L1 scale.
    No environment, neural network, or author benchmark is reproduced here.
    """
    size: int
    eta: float = 0.5
    q: float = 0.72  # gamma*lambda, must be in [0, 1)
    beta2: float = 0.999
    beta_clip: float = 0.9998
    beta_norm: float = 0.9998
    clip_mult: float = 20.0
    policy: bool = False
    t: int = 0
    sigma: float = 0.0
    delta_sq: float = 0.0
    delta_abs: float = 0.0
    v: list = field(default_factory=list)
    e: list = field(default_factory=list)

    def __post_init__(self):
        if self.size < 1 or not 0 <= self.q < 1:
            raise ValueError('positive size and 0 <= q < 1 required')
        if any(not 0 <= b < 1 for b in (self.beta2, self.beta_clip, self.beta_norm)):
            raise ValueError('EMA decays must lie in [0,1)')
        self.v, self.e = [0.0] * self.size, [0.0] * self.size

    def step(self, g, delta, reset=False):
        if len(g) != self.size or not all(math.isfinite(x) for x in [*g, delta]):
            raise ValueError('finite gradient with declared dimension required')
        self.t += 1
        self.v = [self.beta2 * v + (1 - self.beta2) * gi * gi
                  for v, gi in zip(self.v, g)]
        rho = [1 / (math.sqrt(v / (1 - self.beta2 ** self.t)) + 1e-8)
               for v in self.v]
        sigma_now = sum(gi * gi * ri for gi, ri in zip(g, rho))
        self.e = [self.q * e + gi for e, gi in zip(self.e, g)]
        self.sigma = self.q * self.sigma + (1 - self.q) * sigma_now
        sigma_bc = self.sigma / (1 - self.q ** self.t)
        trace_norm = sum(e * e * ri for e, ri in zip(self.e, rho))
        alpha = self.eta / max(math.sqrt(sigma_bc * trace_norm), 1e-8)
        self.delta_sq = self.beta_clip * self.delta_sq + (1 - self.beta_clip) * delta**2
        cap = self.clip_mult * math.sqrt(self.delta_sq / (1 - self.beta_clip ** self.t))
        safe = math.copysign(min(abs(delta), cap), delta)
        if self.policy:
            self.delta_abs = self.beta_norm * self.delta_abs + (1 - self.beta_norm) * abs(safe)
            norm = self.delta_abs / (1 - self.beta_norm ** self.t)
            safe /= max(norm, 1e-12)
        update = [alpha * safe * ri * e for ri, e in zip(rho, self.e)]
        if reset:
            self.e = [0.0] * self.size
        return dict(update=update, alpha=alpha, safe_delta=safe, sigma=sigma_bc)


def intentional_td0(w, x, reward, next_x, gamma, eta):
    """Unpreconditioned, unclipped TD(0): freeze the old bootstrap target."""
    target = reward + gamma * dot(w, next_x)
    delta = target - dot(w, x)
    norm_sq = dot(x, x)
    if norm_sq == 0:
        return list(w), target, delta
    new_w = [wi + eta * delta * xi / norm_sq for wi, xi in zip(w, x)]
    return new_w, target, delta
# END intentional


def metatrace_demo():
    agent = LinearMetatrace(mu=0, gamma=0, lam=0)
    for t in range(2):
        print('fixed-alpha', t + 1, agent.step(1, 2, 0))
    agent = LinearMetatrace(mu=0, gamma=0.9, lam=0.8)
    for t, transition in enumerate([(1, 1, 0.5), (0.5, -0.2, 1), (1, 0.3, 0)]):
        row = agent.step(*transition)
        print('trace', t + 1, row)


def intentional_demo():
    w, x, next_x = [0.5], [2.0], [1.0]
    new, target, delta = intentional_td0(w, x, 1.5, next_x, 0.9, 0.4)
    print('old_delta=', delta, 'new_w=', new,
          'frozen_target_error=', target - dot(new, x),
          'recomputed_td_error=', 1.5 + 0.9 * dot(new, next_x) - dot(new, x))
    opt = IntentionalStep(2)
    for g, d in [([1, 2], 1), ([2, -1], -0.5), ([0.5, 1], 4)]:
        print('normalized-step', opt.step(g, d))


class Checks(unittest.TestCase):
    def test_metatrace_finite_difference(self):
        data = [(1, 1, 0.5), (0.5, -0.2, 1), (1, 0.3, 0)]
        def run(beta):
            a = LinearMetatrace(beta=beta, mu=0)
            for row in data:
                a.step(*row)
            return a
        beta, eps = math.log(0.1), 1e-6
        derivative = (run(beta + eps).w - run(beta - eps).w) / (2 * eps)
        self.assertAlmostEqual(run(beta).h, derivative, places=8)

    def test_metatrace_uses_historical_h(self):
        a = LinearMetatrace(mu=0, gamma=0.5, lam=1)
        a.step(1, 1, 0)
        row = a.step(1, 1, 0)
        self.assertAlmostEqual(row['z_beta'], 0.1)
        # dot(z_t, h_t) would be 1.5*0.1 before the second update, not 0.1.
        self.assertNotAlmostEqual(row['z_beta'], 0.15)

    def test_terminal_uses_past_trace_before_reset(self):
        a = LinearMetatrace(mu=0)
        a.step(1, 1, 1)
        row = a.step(0, 2, 99, terminal=True)
        self.assertAlmostEqual(row['delta'], 2)
        self.assertGreater(row['z'], 0)
        self.assertEqual((a.z, a.z_beta), (0, 0))

    def test_two_step_hand_example(self):
        a = LinearMetatrace(mu=0, gamma=0)
        a.step(1, 2, 0)
        a.step(1, 2, 0)
        self.assertAlmostEqual(a.w, 0.38)
        self.assertAlmostEqual(a.h, 0.36)

    def test_td0_frozen_target_contraction(self):
        new, target, delta = intentional_td0([0.5], [2], 1.5, [1], 0.9, 0.4)
        self.assertAlmostEqual(target - dot(new, [2]), 0.6 * delta)
        self.assertNotAlmostEqual(1.5 + 0.9 * new[0] - 2 * new[0], 0.6 * delta)

    def test_td0_positive_feature_rescaling(self):
        # Same scalar prediction represented by x'=c*x, w'=w/c.
        w1, _, _ = intentional_td0([0.5], [2], 1.5, [1], 0.9, 0.4)
        w2, _, _ = intentional_td0([0.05], [20], 1.5, [10], 0.9, 0.4)
        self.assertAlmostEqual(w1[0], 10 * w2[0])

    def test_first_step_value_change(self):
        g, delta, eta = [1, 2], 0.8, 0.3
        row = IntentionalStep(2, eta=eta).step(g, delta)
        self.assertAlmostEqual(dot(g, row['update']), eta * delta)

    def test_zero_gradient_finite(self):
        row = IntentionalStep(2).step([0, 0], 1)
        self.assertEqual(row['update'], [0, 0])
        self.assertTrue(math.isfinite(row['alpha']))

    def test_policy_first_step_normalizes_delta(self):
        row = IntentionalStep(1, eta=0.05, policy=True).step([2], -3)
        self.assertAlmostEqual(row['safe_delta'], -1)
        self.assertAlmostEqual(2 * row['update'][0], -0.05)

    def test_negative_delta_entropy_sign(self):
        # A local actor direction with zero score derivative and dH/dtheta=1.
        # No trace history: the entropy contribution should increase H for
        # either delta sign. The sign is fixed when constructing the gradient.
        score_grad, entropy_grad, entropy_coeff = 0.0, 1.0, 0.1
        for delta in (-3.0, 3.0):
            g = score_grad + entropy_coeff * entropy_grad * math.copysign(1, delta)
            row = IntentionalStep(1, eta=0.05, q=0, policy=True).step([g], delta)
            self.assertAlmostEqual(entropy_grad * row['update'][0], 0.5)
        naive_g = score_grad + entropy_coeff * entropy_grad
        naive = IntentionalStep(1, eta=0.05, q=0, policy=True).step([naive_g], -3)
        self.assertAlmostEqual(entropy_grad * naive['update'][0], -0.5)

    def test_reset_after_update_not_before(self):
        opt = IntentionalStep(1)
        opt.step([1], 1)
        row = opt.step([0], 1, reset=True)
        self.assertGreater(row['update'][0], 0)
        self.assertEqual(opt.e, [0])
        self.assertGreater(opt.v[0], 0)

    def test_clipping_precedes_policy_scale(self):
        opt = IntentionalStep(1, clip_mult=0.5, policy=True)
        row = opt.step([1], 8)
        self.assertAlmostEqual(row['safe_delta'], 1)
        self.assertAlmostEqual(opt.delta_abs / (1 - opt.beta_norm), 4)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            IntentionalStep(1, q=1)
        with self.assertRaises(ValueError):
            IntentionalStep(2).step([1], 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', choices=['all', 'metatrace', 'intentional', 'test'])
    args = parser.parse_args()
    if args.experiment == 'test':
        unittest.main(argv=['meta_frontier_lab'], verbosity=2)
    else:
        if args.experiment in ('all', 'metatrace'):
            metatrace_demo()
        if args.experiment in ('all', 'intentional'):
            intentional_demo()
