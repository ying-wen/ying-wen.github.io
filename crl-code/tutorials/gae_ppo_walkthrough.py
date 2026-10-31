#!/usr/bin/env python3
"""A fixed three-transition GAE/PPO example, not a training experiment.

Standard library only. Run with no arguments for JSON, or --test for assertions.
Equations: GAE (2015), section 3; PPO (2017), sections 3 and 5.
Compare Spinning Up PPOBuffer.finish_path: its critic uses reward-to-go,
whereas this tutorial explicitly chooses the finite lambda-return.
"""
from dataclasses import dataclass, FrozenInstanceError
from fractions import Fraction as F
import json
import math
import sys
import unittest


def advantages(rewards, values, bootstrap, continuation, gamma=F(1), lam=F(1, 2)):
    """values has length T+1; each next value is the FINAL observation value.

    In this small sequential example values[t+1] belongs to the same episode.
    For concatenated episodes pass separate next-values, not reset observations.
    b controls value bootstrap; c controls whether next row's advantage connects.
    """
    n = len(rewards)
    assert len(values) == n + 1 and len(bootstrap) == len(continuation) == n
    delta = tuple(rewards[t] + gamma * bootstrap[t] * values[t+1] - values[t] for t in range(n))
    adv, carry = [F(0)] * n, F(0)
    for t in reversed(range(n)):
        carry = delta[t] + gamma * lam * continuation[t] * carry
        adv[t] = carry
    return delta, tuple(adv)


@dataclass(frozen=True)
class FrozenBatch:
    old_logp: tuple
    raw_advantage: tuple
    critic_target: tuple


def prepare_batch():
    rewards, values = (F(0), F(0), F(1)), (F(1, 5), F(4, 5), F(2, 5), F(0))
    delta, adv = advantages(rewards, values, (1, 1, 0), (1, 1, 0))
    batch = FrozenBatch((math.log(.5),) * 3, adv, tuple(a + v for a, v in zip(adv, values)))
    return rewards, values, delta, batch


def clipped_term(ratio, advantage, epsilon=.2):
    return min(ratio * advantage, min(1 + epsilon, max(1 - epsilon, ratio)) * advantage)


def logp_derivative(ratio, advantage, epsilon=.2):
    """Partial derivative of ONE objective term wrt current log p(a|s).

    Other action probabilities are not independent policy parameters.
    Excludes the two nondifferentiable boundaries; use chain rule afterwards.
    """
    assert not math.isclose(ratio, 1-epsilon) and not math.isclose(ratio, 1+epsilon)
    flat = (advantage > 0 and ratio > 1+epsilon) or (advantage < 0 and ratio < 1-epsilon)
    return 0. if flat else ratio * advantage


def kl(old, new):
    assert math.isclose(sum(old), 1) and math.isclose(sum(new), 1)
    return sum(p * math.log(p/q) for p, q in zip(old, new) if p)


def walkthrough():
    rewards, values, delta, batch = prepare_batch()
    probabilities = (.7, .3, .55)  # a chosen candidate, not the result of training
    ratios = tuple(math.exp(math.log(p)-lp) for p, lp in zip(probabilities, batch.old_logp))
    terms = tuple(clipped_term(r, float(a)) for r, a in zip(ratios, batch.raw_advantage))
    derivatives = tuple(logp_derivative(r, float(a)) for r, a in zip(ratios, batch.raw_advantage))
    # All three Bernoulli logits share an additive parameter w; here w=0.
    # A single EXACT gradient evaluation, no optimizer loop or random rollout.
    shared_gradient = sum(d * (1-p) for d, p in zip(derivatives, probabilities))/3
    shifted = tuple(1/(1+math.exp(-(math.log(p/(1-p)) + .1*shared_gradient))) for p in probabilities)
    # This fixed corridor ends with reward 1 iff all three chosen actions succeed;
    # every alternative action terminates with reward 0. Enumerate exactly.
    old_return, candidate_return = .5**3, math.prod(probabilities)
    return dict(kind="constructed-exact-example", gamma=1, lam=.5, epsilon=.2,
                rewards=list(map(float, rewards)), values=list(map(float, values)),
                delta=list(map(float, delta)), advantage=list(map(float, batch.raw_advantage)),
                lambda_target=list(map(float, batch.critic_target)), reward_to_go=[1., 1., 1.],
                old_probability=[.5]*3, probability=list(probabilities), ratio=list(ratios),
                term=list(terms), logp_derivative=list(derivatives), shared_gradient=shared_gradient,
                shifted_probability=list(shifted), unsampled_action_kl=kl((.5, .25, .25), (.5, .499, .001)),
                exact_evaluation=dict(old_return=old_return, candidate_return=candidate_return),
                random_rollouts=0, optimizer_steps=0, hypothetical_analytic_steps=1)


class WalkthroughTests(unittest.TestCase):
    def test_exact_fractions_and_target_choice(self):
        _, _, delta, b = prepare_batch()
        self.assertEqual(delta, (F(3,5), F(-2,5), F(3,5)))
        self.assertEqual(b.raw_advantage, (F(11,20), F(-1,10), F(3,5)))
        self.assertEqual(b.critic_target, (F(3,4), F(7,10), F(1)))
        self.assertNotEqual(b.critic_target, (F(1),)*3)

    def test_terminal_truncation_and_lambda_extremes(self):
        self.assertEqual(advantages((F(1),), (F(2,5), F(9)), (0,), (0,))[1], (F(3,5),))
        self.assertEqual(advantages((F(0),), (F(2,5), F(3,10)), (1,), (0,))[1], (F(-1,10),))
        r, v, d, _ = prepare_batch()
        self.assertEqual(advantages(r, v, (1,1,0), (1,1,0), lam=F(0))[1], d)
        self.assertEqual(advantages(r, v, (1,1,0), (1,1,0), lam=F(1))[1], (F(4,5), F(1,5), F(3,5)))

    def test_frozen_quantities_and_normalization(self):
        _, _, _, b = prepare_batch()
        with self.assertRaises(FrozenInstanceError):
            b.old_logp = (0, 0, 0)
        snapshot = (b.old_logp, b.raw_advantage, b.critic_target)
        for p in (.4, .55, .7):
            clipped_term(math.exp(math.log(p)-b.old_logp[0]), float(b.raw_advantage[0]))
        self.assertEqual(snapshot, (b.old_logp, b.raw_advantage, b.critic_target))
        adv = list(map(float, b.raw_advantage))
        mean = sum(adv)/3
        std = math.sqrt(sum((a-mean)**2 for a in adv)/3)
        actor_adv = [(a-mean)/std for a in adv]
        self.assertAlmostEqual(sum(actor_adv), 0)
        self.assertEqual(b.critic_target, (F(3,4), F(7,10), F(1)))

    def test_all_sign_branches_and_finite_differences(self):
        for a in (.55, -.1, 0.):
            for r in (.6, 1., 1.1, 1.4):
                h = 1e-6
                fd = (clipped_term(r*math.exp(h), a)-clipped_term(r*math.exp(-h), a))/(2*h)
                self.assertAlmostEqual(fd, logp_derivative(r, a), places=8)
        self.assertAlmostEqual(clipped_term(.6, .55), .33)  # harmful direction NOT flat
        self.assertAlmostEqual(clipped_term(1.4, -.1), -.14)

    def test_shared_parameter_and_unsampled_actions(self):
        d = walkthrough()
        self.assertAlmostEqual(d['shared_gradient'], .099)
        self.assertGreater(d['shifted_probability'][0]/.5, 1.4)
        self.assertEqual(.5/.5, 1.)  # sampled action ratio unchanged
        self.assertGreater(d['unsampled_action_kl'], 1.)

    def test_surrogate_is_not_independent_evaluation(self):
        d = walkthrough()
        self.assertGreater(sum(d['term']), sum(d['advantage']))
        self.assertAlmostEqual(d['exact_evaluation']['old_return'], .125)
        self.assertAlmostEqual(d['exact_evaluation']['candidate_return'], .1155)
        self.assertLess(d['exact_evaluation']['candidate_return'], d['exact_evaluation']['old_return'])


if __name__ == '__main__':
    if '--test' in sys.argv:
        unittest.main(argv=[sys.argv[0]])
    else:
        print(json.dumps(walkthrough(), indent=2))
