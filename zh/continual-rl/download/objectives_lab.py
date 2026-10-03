"""Exact, standard-library experiments on reinforcement-learning objectives.

Run: python3 objectives_lab.py all | preferences | stopping | shaping | lifetime | test
Code: MIT. These finite calculations illustrate objective definitions; they are
not benchmark results or implementations of an author's learning algorithm.
"""
import argparse
import math
import unittest


def finite(values, name):
    result = tuple(float(x) for x in values)
    if not all(math.isfinite(x) for x in result):
        raise ValueError(name + ' must contain only finite numbers')
    return result


def discount(gamma, continuing=False):
    if not math.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError('gamma must be in [0, 1]')
    if continuing and gamma == 1:
        raise ValueError('an infinite discounted sum requires gamma < 1')


# BEGIN returns
def discounted_return(rewards, gamma, bootstrap=0.0):
    """Sum gamma**t * rewards[t], plus gamma**T * bootstrap.

    bootstrap=0 describes a true terminal boundary. For a truncated continuing
    trajectory, the caller can supply its continuation value instead.
    """
    discount(gamma)
    rewards = finite(rewards, 'rewards')
    if not math.isfinite(bootstrap):
        raise ValueError('bootstrap must be finite')
    value = float(bootstrap)
    for reward in reversed(rewards):
        value = reward + gamma * value
    return value


def periodic_value(cycle, gamma, phase=0):
    """Exact value of an infinite deterministic reward cycle at a given phase."""
    discount(gamma, continuing=True)
    cycle = finite(cycle, 'cycle')
    if not cycle:
        raise ValueError('cycle must be non-empty')
    if not isinstance(phase, int):
        raise ValueError('phase must be an integer')
    rewards = cycle[phase % len(cycle):] + cycle[:phase % len(cycle)]
    return discounted_return(rewards, gamma) / (1 - gamma ** len(cycle))


def periodic_average_and_bias(cycle):
    """Solve g+h(i)=r(i)+h(i+1) with h(0)=0, including periodic chains."""
    cycle = finite(cycle, 'cycle')
    if not cycle:
        raise ValueError('cycle must be non-empty')
    rate = sum(cycle) / len(cycle)
    bias = [0.0]
    for reward in cycle[:-1]:
        bias.append(bias[-1] + rate - reward)
    return rate, bias


def prefix_rewards(cycle, steps):
    cycle = finite(cycle, 'cycle')
    if not cycle or not isinstance(steps, int) or steps < 0:
        raise ValueError('non-empty cycle and nonnegative integer steps required')
    return [cycle[t % len(cycle)] for t in range(steps)]
# END returns


# BEGIN stopping
def stopped_return_exact(rewards, gamma):
    """Expected undiscounted sum with an independent geometric stopping time.

    Collect the first reward for sure, then survive after each reward with
    probability gamma. Rewards beyond the supplied list are zero. The last
    outcome combines all stopping times at or beyond the end of the list.
    """
    discount(gamma)
    rewards = finite(rewards, 'rewards')
    expectation = 0.0
    partial_sum = 0.0
    for index, reward in enumerate(rewards):
        partial_sum += reward
        survival = gamma ** index
        probability = survival if index == len(rewards) - 1 else (1 - gamma) * survival
        expectation += probability * partial_sum
    return expectation
# END stopping


# BEGIN shaping
def shape_rewards(rewards, potentials, gamma):
    """F_t=gamma*Phi(s[t+1])-Phi(s[t]); retain the final potential explicitly."""
    discount(gamma)
    rewards = finite(rewards, 'rewards')
    potentials = finite(potentials, 'potentials')
    if len(potentials) != len(rewards) + 1:
        raise ValueError('one potential per state, including both endpoints, required')
    return [reward + gamma * potentials[t + 1] - potentials[t]
            for t, reward in enumerate(rewards)]


def shaping_comparison(rewards, potentials, gamma):
    shaped = shape_rewards(rewards, potentials, gamma)
    original = discounted_return(rewards, gamma)
    boundary = -potentials[0] + gamma ** len(rewards) * potentials[-1]
    return dict(original=original, shaped=discounted_return(shaped, gamma),
                boundary=boundary, expected=original + boundary)
# END shaping


# BEGIN lifetime
def evaluate_schedule(arm_rewards, action_schedule):
    """Compare online reward with the value of freezing the final bandit action.

    This is an evaluator of prescribed action schedules, not a bandit learner.
    The stationary deterministic environment returns arm_rewards[action].
    """
    arm_rewards = finite(arm_rewards, 'arm_rewards')
    actions = tuple(action_schedule)
    if not arm_rewards or not actions:
        raise ValueError('non-empty arms and action schedule required')
    if any(not isinstance(a, int) or not 0 <= a < len(arm_rewards) for a in actions):
        raise ValueError('each action must index an arm')
    rewards = [arm_rewards[a] for a in actions]
    cumulative = sum(rewards)
    return dict(cumulative=cumulative, online_mean=cumulative / len(rewards),
                frozen_final_mean=arm_rewards[actions[-1]])
# END lifetime


def preferences_demo():
    a, b = (1,), (0, 0, 4)
    print('discount preference switches at gamma=', (1 + math.sqrt(13)) / 6)
    for gamma in (0.5, 0.9):
        print('gamma=', gamma, 'A=', periodic_value(a, gamma),
              'B=', periodic_value(b, gamma))
    for label, cycle in [('A', a), ('B', b)]:
        rate, bias = periodic_average_and_bias(cycle)
        print(label, 'average=', rate, 'bias=', bias,
              'first_5_reward=', sum(prefix_rewards(cycle, 5)))


def stopping_demo():
    rewards, gamma = [1, 2, 3], 0.5
    print('discounted=', discounted_return(rewards, gamma),
          'independent_stopping=', stopped_return_exact(rewards, gamma))
    print('one observed reward with a continuing tail:',
          discounted_return([1], 0.9, bootstrap=10),
          'same prefix treated as terminal:', discounted_return([1], 0.9))


def shaping_demo():
    print('zero-terminal potential:', shaping_comparison([0, 2], [3, 1, 0], 0.9))
    print('action A, valid terminal:', shaping_comparison([1], [0, 0], 0.9))
    print('action B, nonzero terminal:', shaping_comparison([0], [0, 2], 0.9))


def lifetime_demo():
    arms = [0, 0.6, 1]
    print('early moderate policy:', evaluate_schedule(arms, [1] * 10))
    print('late excellent policy:', evaluate_schedule(arms, [0] * 8 + [2] * 2))


class Checks(unittest.TestCase):
    def test_discounted_hand_calculation(self):
        self.assertAlmostEqual(discounted_return([1, 2, 3], 0.5), 2.75)

    def test_discount_zero_and_one(self):
        self.assertEqual(discounted_return([1, 2, 3], 0), 1)
        self.assertEqual(discounted_return([1, 2, 3], 1), 6)

    def test_periodic_bellman_equations(self):
        rewards, gamma = (0, 0, 4), 0.9
        values = [periodic_value(rewards, gamma, i) for i in range(3)]
        for i, reward in enumerate(rewards):
            self.assertAlmostEqual(values[i], reward + gamma * values[(i + 1) % 3])

    def test_discount_changes_policy_preference(self):
        self.assertGreater(periodic_value((1,), 0.5), periodic_value((0, 0, 4), 0.5))
        self.assertLess(periodic_value((1,), 0.9), periodic_value((0, 0, 4), 0.9))
        threshold = (1 + math.sqrt(13)) / 6
        self.assertAlmostEqual(periodic_value((1,), threshold),
                               periodic_value((0, 0, 4), threshold))

    def test_average_differs_from_short_discount(self):
        rate, bias = periodic_average_and_bias((0, 0, 4))
        self.assertAlmostEqual(rate, 4 / 3)
        self.assertGreater(rate, periodic_average_and_bias((1,))[0])
        for i, reward in enumerate((0, 0, 4)):
            self.assertAlmostEqual(rate + bias[i], reward + bias[(i + 1) % 3])

    def test_normalized_discount_limit(self):
        gamma = 1 - 1e-7
        self.assertAlmostEqual((1 - gamma) * periodic_value((0, 0, 4), gamma),
                               4 / 3, places=6)

    def test_phase_and_finite_lifetime_matter(self):
        self.assertGreater(periodic_value((0, 0, 4), 0.5, 2),
                           periodic_value((0, 0, 4), 0.5, 0))
        self.assertEqual(sum(prefix_rewards((0, 0, 4), 5)), 4)
        self.assertEqual(sum(prefix_rewards((1,), 5)), 5)

    def test_independent_stopping_identity(self):
        for rewards in ([], [2], [1, -2, 4, 3]):
            for gamma in (0, 0.2, 0.9, 1):
                self.assertAlmostEqual(stopped_return_exact(rewards, gamma),
                                       discounted_return(rewards, gamma))

    def test_reward_correlated_stopping_counterexample(self):
        # First reward is zero. A fair hidden outcome decides BOTH continuation
        # and the next reward: continue exactly when that reward would be two.
        actual_second_reward = 0.5 * 2 + 0.5 * 0
        wrong_independence_factorization = 0.5 * (0.5 * 2 + 0.5 * 0)
        self.assertEqual(actual_second_reward, 1)
        self.assertEqual(wrong_independence_factorization, 0.5)

    def test_timeout_is_not_terminal(self):
        self.assertAlmostEqual(discounted_return([1], 0.9, bootstrap=10), 10)
        self.assertEqual(discounted_return([1], 0.9), 1)

    def test_shaping_telescopes(self):
        for gamma in (0, 0.5, 1):
            result = shaping_comparison([1, -2, 4], [3, 1, -1, 2], gamma)
            self.assertAlmostEqual(result['shaped'], result['expected'])

    def test_zero_terminal_preserves_policy_ranking(self):
        gamma = 0.9
        a = shaping_comparison([1], [3, 0], gamma)
        b = shaping_comparison([0, 2], [3, 1, 0], gamma)
        self.assertGreater(b['original'], a['original'])
        self.assertGreater(b['shaped'], a['shaped'])
        self.assertAlmostEqual(a['boundary'], b['boundary'])

    def test_bad_terminal_potential_reverses_ranking(self):
        a = shaping_comparison([1], [0, 0], 0.9)
        b = shaping_comparison([0], [0, 2], 0.9)
        self.assertGreater(a['original'], b['original'])
        self.assertLess(a['shaped'], b['shaped'])

    def test_equal_nonzero_terminal_with_variable_duration(self):
        a = shaping_comparison([1], [0, 10], 0.9)
        b = shaping_comparison([0, 1.5], [0, 0, 10], 0.9)
        self.assertLess(a['original'], b['original'])
        self.assertGreater(a['shaped'], b['shaped'])

    def test_continuing_shaping_with_bootstrap(self):
        gamma, phi = 0.9, [3, 2]
        original = discounted_return([1], gamma, bootstrap=10)
        shaped_reward = shape_rewards([1], phi, gamma)
        shaped = discounted_return(shaped_reward, gamma, bootstrap=10 - phi[-1])
        self.assertAlmostEqual(shaped, original - phi[0])

    def test_average_shaping_cancels_on_cycle(self):
        rewards, potentials = [0, 0, 4], [3, -2, 1, 3]
        self.assertAlmostEqual(sum(shape_rewards(rewards, potentials, 1)), sum(rewards))

    def test_constant_reward_shift_can_change_episodic_choice(self):
        a, b = [2], [0, 0, 3]
        self.assertLess(sum(a), sum(b))
        self.assertGreater(sum(r - 1 for r in a), sum(r - 1 for r in b))

    def test_lifetime_vs_frozen_ranking(self):
        a = evaluate_schedule([0, 0.6, 1], [1] * 10)
        b = evaluate_schedule([0, 0.6, 1], [0] * 8 + [2] * 2)
        self.assertGreater(a['cumulative'], b['cumulative'])
        self.assertLess(a['frozen_final_mean'], b['frozen_final_mean'])

    def test_input_validation(self):
        for gamma in (-0.1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                discounted_return([1], gamma)
        with self.assertRaises(ValueError):
            periodic_value([1], 1)
        with self.assertRaises(ValueError):
            periodic_average_and_bias([])
        with self.assertRaises(ValueError):
            shape_rewards([1], [0], 0.9)
        with self.assertRaises(ValueError):
            evaluate_schedule([1], [1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['all', 'preferences', 'stopping', 'shaping', 'lifetime', 'test'])
    args = parser.parse_args()
    demos = dict(preferences=preferences_demo, stopping=stopping_demo,
                 shaping=shaping_demo, lifetime=lifetime_demo)
    if args.command == 'test':
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
        raise SystemExit(0 if result.wasSuccessful() else 1)
    for name, demo in demos.items():
        if args.command in ('all', name):
            print(name)
            demo()


if __name__ == '__main__':
    main()
