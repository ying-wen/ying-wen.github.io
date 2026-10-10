#!/usr/bin/env python3
"""Two-step position control: fixed critic errors and SAC pathwise derivatives.

Standard library only; JSON by default, --test for independent assertions.
No sampled rollout, fitted critic or training benchmark is performed here.
State=(x, remaining_steps), action in [-2, 2], x'=x+a,
reward=-(x'-1)^2-.1*a^2, termination after two actions, gamma=.9.
The final-step Q is exactly the reward, for both ordinary and soft Q conventions.
"""
import json
import math
import sys
import unittest


def reward(x, a):
    return -(x + a - 1) ** 2 - .1 * a * a


def transition(x, remaining, a):
    if remaining < 1 or not -2 <= a <= 2:
        raise ValueError("a live state and action in [-2,2] are required")
    return x + a, remaining - 1, reward(x, a), remaining == 1


def q_last(a, x=.25):
    return reward(x, a)


def q_slope(a, x=.25):
    return 2 * (1 - x) - 2.2 * a


def critic(a, amplitude=.8):
    return q_last(a) + amplitude * math.exp(-.5 * ((a - 1.4) / .12) ** 2)


def critic_slope(a, amplitude=.8):
    bump = amplitude * math.exp(-.5 * ((a - 1.4) / .12) ** 2)
    return q_slope(a) + bump * (1.4 - a) / .12 ** 2


def target_action(base, noise, cap=.2):
    return min(2, max(-2, base + min(cap, max(-cap, noise))))


def simpson(f, left, right, intervals=2000):
    if intervals <= 0 or intervals % 2:
        raise ValueError("Simpson intervals must be positive and even")
    h = (right - left) / intervals
    total = f(left) + f(right)
    for i in range(1, intervals):
        total += (4 if i % 2 else 2) * f(left + h * i)
    return total * h / 3


def smooth_value(base=1.4, sigma=.15, cap=.2, intervals=2000):
    """Clipped Normal includes atoms at +/-cap, not a truncated Normal.

    Expected min of the two fixed target critics; deterministic quadrature.
    Their common bump has amplitudes .8 and .3, so critic 2 is always min.
    """
    if sigma <= 0 or cap <= 0:
        raise ValueError("positive sigma and cap required")
    tail = .5 * math.erfc(cap / (sigma * math.sqrt(2)))
    f = lambda e: critic(target_action(base, e, cap), .3)
    density = lambda e: math.exp(-.5*(e/sigma)**2) / (sigma*math.sqrt(2*math.pi))
    return (tail * (f(-cap) + f(cap))
            + simpson(lambda e: f(e)*density(e), -cap, cap, intervals))


def log_jacobian(u):
    # log(1-tanh(u)^2), without subtracting nearly equal floats.
    v = abs(u)
    return 2 * (math.log(2) - v - math.log1p(math.exp(-2*v)))


def sac_sample(mean=.3, log_std=-.7, epsilon=.4, scale=2, alpha=.2):
    if scale <= 0:
        raise ValueError("positive physical action scale required")
    sigma = math.exp(log_std)
    u = mean + sigma * epsilon
    z = math.tanh(u)
    a = scale * z
    # Substitute u=mean+sigma*epsilon BEFORE differentiating the sampled loss.
    lp_normal = -.5*epsilon**2 - log_std - .5*math.log(2*math.pi)
    lp_norm = lp_normal - log_jacobian(u)
    lp = lp_norm - math.log(scale)
    da_dm = scale * (1-z*z)
    noise_path = sigma * epsilon
    entropy_mean = alpha * 2*z
    value_mean = -q_slope(a) * da_dm
    entropy_log_std = alpha * (-1 + 2*z*noise_path)
    value_log_std = value_mean * noise_path
    return dict(mean=mean, log_std=log_std, epsilon=epsilon, sigma=sigma, u=u,
                normalized_action=z, action=a, logp_normal=lp_normal,
                log_jacobian=log_jacobian(u), logp_normalized=lp_norm, logp=lp,
                q=q_last(a), loss=alpha*lp-q_last(a), da_dmean=da_dm,
                gradient_mean=entropy_mean+value_mean,
                gradient_log_std=entropy_log_std+value_log_std,
                entropy_mean=entropy_mean, value_mean=value_mean,
                entropy_log_std=entropy_log_std, value_log_std=value_log_std)


def walkthrough():
    xp, remaining, r, terminal = transition(0, 2, .25)
    a = 1.2
    theta = math.atanh(a/2)
    jac = 2*(1-(a/2)**2)
    g = critic_slope(a)*jac
    next_a = 2*math.tanh(theta+.05*g)
    smoothed = smooth_value()
    s = sac_sample()
    target_entropy_norm = .1
    target_entropy_env = target_entropy_norm + math.log(2)
    residual = s['logp'] + target_entropy_env
    return dict(
        kind='constructed-numerical-example', random_rollouts=0, optimizer_steps=0,
        hypothetical_analytic_steps=1,
        task=dict(initial=[0,2], observed_action=.25, next_state=[xp,remaining],
                  reward=r, terminal=terminal, gamma=.9, action_bounds=[-2,2]),
        actor=dict(action=a, theta=theta, da_dtheta=jac, true_slope=q_slope(a),
                   critic_slope=critic_slope(a), gradient=g, step_size=.05,
                   next_action=next_a, true_before=q_last(a), true_after=q_last(next_a),
                   estimated_before=critic(a), estimated_after=critic(next_a)),
        targets=dict(base_action=1.4, sigma=.15, noise_clip=.2,
                     q1_center=critic(1.4), q2_center=critic(1.4,.3),
                     min_smoothed=smoothed, ddpg=r+.9*critic(1.4),
                     twin_only=r+.9*critic(1.4,.3), td3_expectation=r+.9*smoothed,
                     gaussian_tail_mass=.5*math.erfc(.2/(.15*math.sqrt(2))),
                     terminal_target=r, quadrature_intervals=2000),
        sac=s | dict(alpha=.2, target=r+.9*(s['q']-.2*s['logp']),
                     target_entropy_normalized=target_entropy_norm,
                     target_entropy_physical=target_entropy_env,
                     temperature_residual=residual, beta_gradient=-.2*residual,
                     beta_gradient_unshifted=-.2*(s['logp']+target_entropy_norm)))


def difference(f, x, h=1e-6):
    return (f(x+h)-f(x-h))/(2*h)


class WalkthroughTests(unittest.TestCase):
    def test_task_and_terminal(self):
        self.assertEqual(transition(0,2,.25), (.25,1,-.56875,False))
        x,h,r,d = transition(.25,1,1.2)
        self.assertEqual((h,d), (0,True))
        self.assertAlmostEqual(r,-.3465)
        with self.assertRaises(ValueError):
            transition(0,0,0)

    def test_actor_gradient_and_actual_reward(self):
        d = walkthrough()['actor']
        numerical = difference(lambda th: critic(2*math.tanh(th)),d['theta'])
        self.assertAlmostEqual(d['gradient'], numerical, places=7)
        self.assertLess(d['true_slope'],0)
        self.assertGreater(d['critic_slope'],0)
        self.assertGreater(d['estimated_after'],d['estimated_before'])
        self.assertLess(d['true_after'],d['true_before'])

    def test_clipping_and_gaussian_atoms(self):
        self.assertEqual(target_action(1.9,3),2)
        self.assertEqual(target_action(-1.9,-3),-2)
        tail=.5*math.erfc(.2/(.15*math.sqrt(2)))
        mass=simpson(lambda e: math.exp(-.5*(e/.15)**2)/(.15*math.sqrt(2*math.pi)),-.2,.2)
        self.assertAlmostEqual(mass+2*tail,1,places=11)
        self.assertAlmostEqual(smooth_value(intervals=1000),smooth_value(intervals=2000),places=10)
        d=walkthrough()['targets']
        self.assertLess(d['td3_expectation'],d['twin_only'])
        self.assertLess(d['twin_only'],d['ddpg'])

    def test_reparameterization_both_parameters(self):
        s=sac_sample()
        self.assertAlmostEqual(s['gradient_mean'],difference(lambda m: sac_sample(mean=m)['loss'],.3),places=7)
        self.assertAlmostEqual(s['gradient_log_std'],difference(lambda l: sac_sample(log_std=l)['loss'],-.7),places=7)
        self.assertNotAlmostEqual(s['gradient_mean'],s['entropy_mean'],places=3)
        self.assertAlmostEqual(log_jacobian(.5),math.log(1-math.tanh(.5)**2),places=12)
        self.assertTrue(math.isfinite(log_jacobian(40)))

    def test_density_and_temperature_coordinates(self):
        s=walkthrough()['sac']
        self.assertAlmostEqual(s['logp_normalized']-s['logp'],math.log(2))
        self.assertAlmostEqual(s['temperature_residual'],s['logp_normalized']+.1)
        self.assertLess(s['beta_gradient'],0)
        self.assertGreater(s['beta_gradient_unshifted'],0)
        b=math.log(.2)
        loss=lambda beta: -math.exp(beta)*s['temperature_residual']
        self.assertAlmostEqual(s['beta_gradient'],difference(loss,b),places=9)
        # Integrate the physical density through u; preserve probability mass.
        sigma=math.exp(-.7)
        def mass(u):
            eps=(u-.3)/sigma
            lp=-.5*eps*eps+.7-.5*math.log(2*math.pi)-log_jacobian(u)-math.log(2)
            return math.exp(lp+log_jacobian(u)+math.log(2))
        self.assertAlmostEqual(simpson(mass,.3-8*sigma,.3+8*sigma),1,places=11)


if __name__ == '__main__':
    if '--test' in sys.argv:
        unittest.main(argv=[sys.argv[0]])
    else:
        print(json.dumps(walkthrough(),indent=2,ensure_ascii=False))
