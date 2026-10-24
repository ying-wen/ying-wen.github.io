"""Finite discounted control: exact planning and a genuine interaction loop.

Python 3.10+, standard library only. This is a teaching MDP, not a benchmark
reproduction. DP receives the model; Q-learning sees only sampled transitions.
Reported greedy values are evaluated afterwards with the model, separately
from the rewards earned by the exploratory training policy.
"""
import argparse
import random
import unittest


# BEGIN control_mdp
GAMMA = 0.9
# Outcomes are (probability, reward, next_state). State 2 is terminal.
MDP = {
    0: {0: [(1.0, 1.0, 2)], 1: [(1.0, 0.0, 1)]},
    1: {0: [(1.0, 2.0, 2)], 1: [(1.0, 0.5, 0)]},
    2: {},
}


def action_value(state, action, values, gamma=GAMMA):
    return sum(p * (r + gamma * values[nxt])
               for p, r, nxt in MDP[state][action])


def greedy_policy(values, gamma=GAMMA):
    # Deterministic, consistent tie breaking for finite policy iteration.
    return {s: max(actions, key=lambda a: action_value(s, a, values, gamma))
            for s, actions in MDP.items() if actions}


def sample_transition(state, action, rng):
    draw, cumulative = rng.random(), 0.0
    for p, reward, nxt in MDP[state][action]:
        cumulative += p
        if draw < cumulative:
            return reward, nxt, not MDP[nxt]
    raise ValueError("Transition probabilities must sum to one")
# END control_mdp


# BEGIN dynamic_programming
def evaluate_policy(policy, gamma=GAMMA, tolerance=1e-12):
    """Synchronous expectation backups; terminal value stays zero."""
    values = {s: 0.0 for s in MDP}
    for _ in range(100000):
        updated = {s: action_value(s, policy[s], values, gamma)
                   if actions else 0.0 for s, actions in MDP.items()}
        change = max(abs(updated[s] - values[s]) for s in MDP)
        values = updated
        if change < tolerance:
            return values
    raise RuntimeError("Policy evaluation did not converge")


def policy_iteration():
    policy, history = {0: 0, 1: 0}, []
    while True:
        values = evaluate_policy(policy)
        history.append((policy.copy(), values.copy()))
        improved = greedy_policy(values)
        if improved == policy:
            return policy, values, history
        policy = improved


def optimality_backup(values, gamma=GAMMA):
    return {s: max(action_value(s, a, values, gamma) for a in actions)
            if actions else 0.0 for s, actions in MDP.items()}


def value_iteration(tolerance=1e-10):
    values = {s: 0.0 for s in MDP}
    for sweep in range(1, 100000):
        values = optimality_backup(values)
        next_values = optimality_backup(values)
        residual = max(abs(next_values[s] - values[s]) for s in MDP)
        if residual <= tolerance:
            return greedy_policy(values), values, sweep, residual
    raise RuntimeError("Value iteration did not converge")
# END dynamic_programming


# BEGIN sampled_control
def epsilon_probabilities(row, epsilon):
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be between zero and one")
    best = max(row.values())
    ties = [a for a, value in row.items() if value == best]
    return {a: epsilon / len(row) +
            ((1.0 - epsilon) / len(ties) if a in ties else 0.0)
            for a in row}


def choose_action(row, epsilon, rng):
    probs = epsilon_probabilities(row, epsilon)
    draw, cumulative = rng.random(), 0.0
    for action, probability in probs.items():
        cumulative += probability
        if draw < cumulative:
            return action
    return next(reversed(probs))  # floating-point rounding only


def q_learning(steps=20000, epsilon=0.2, alpha=0.1, seed=7):
    rng = random.Random(seed)
    q = {s: {a: 0.0 for a in actions}
         for s, actions in MDP.items() if actions}
    visits = {s: {a: 0 for a in row} for s, row in q.items()}
    state, reward_sum, resets = 0, 0.0, 0
    for _ in range(steps):
        # This behavior policy changes whenever Q changes.
        action = choose_action(q[state], epsilon, rng)
        reward, nxt, terminated = sample_transition(state, action, rng)
        target = reward if terminated else reward + GAMMA * max(q[nxt].values())
        q[state][action] += alpha * (target - q[state][action])
        visits[state][action] += 1
        reward_sum += reward
        if terminated:
            resets += 1
            state = 0  # A new task episode, only after a true terminal state.
        else:
            state = nxt
    policy = {s: max(row, key=row.get) for s, row in q.items()}
    return q, policy, visits, reward_sum, resets
# END sampled_control


class ControlTests(unittest.TestCase):
    def test_transition_probabilities(self):
        for actions in MDP.values():
            for outcomes in actions.values():
                self.assertAlmostEqual(sum(p for p, _, _ in outcomes), 1.0)

    def test_initial_policy(self):
        self.assertEqual(evaluate_policy({0: 0, 1: 0}), {0: 1.0, 1: 2.0, 2: 0.0})

    def test_one_policy_improvement(self):
        self.assertEqual(greedy_policy(evaluate_policy({0: 0, 1: 0})), {0: 1, 1: 0})

    def test_policy_iteration(self):
        policy, values, history = policy_iteration()
        self.assertEqual(policy, {0: 1, 1: 1})
        self.assertEqual(len(history), 3)
        self.assertAlmostEqual(values[0], 45 / 19, places=9)
        self.assertAlmostEqual(values[1], 50 / 19, places=9)

    def test_policy_improvement_is_monotone(self):
        _, _, history = policy_iteration()
        for (_, old), (_, new) in zip(history, history[1:]):
            for state in MDP:
                self.assertGreaterEqual(new[state] + 1e-11, old[state])

    def test_value_iteration_and_residual_certificate(self):
        policy, values, _, residual = value_iteration()
        error = max(abs(values[0] - 45 / 19), abs(values[1] - 50 / 19))
        self.assertEqual(policy, {0: 1, 1: 1})
        self.assertLessEqual(error, residual / (1 - GAMMA) + 1e-14)

    def test_terminal_reward_is_not_dropped(self):
        self.assertEqual(action_value(1, 0, {0: 100, 1: 100, 2: 0}), 2.0)
        self.assertEqual(sample_transition(1, 0, random.Random(1)), (2.0, 2, True))

    def test_gamma_zero_selects_immediate_reward(self):
        self.assertEqual(greedy_policy({s: 0 for s in MDP}, gamma=0), {0: 0, 1: 0})

    def test_epsilon_greedy_ties(self):
        self.assertEqual(epsilon_probabilities({0: 0, 1: 0}, 0.2), {0: 0.5, 1: 0.5})
        probs = epsilon_probabilities({0: 1, 1: 2}, 0.2)
        self.assertAlmostEqual(probs[0], 0.1)
        self.assertAlmostEqual(probs[1], 0.9)

    def test_interactive_q_learning(self):
        q, policy, visits, _, resets = q_learning()
        self.assertEqual(policy, {0: 1, 1: 1})
        self.assertGreater(resets, 0)
        self.assertTrue(all(count > 100 for row in visits.values() for count in row.values()))
        self.assertAlmostEqual(q[0][1], 45 / 19, places=8)
        self.assertAlmostEqual(q[1][1], 50 / 19, places=8)

    def test_single_step_cannot_learn_unvisited_action(self):
        q, _, visits, _, _ = q_learning(steps=1)
        self.assertEqual(sum(sum(row.values()) for row in visits.values()), 1)
        for state, row in visits.items():
            for action, count in row.items():
                if count == 0:
                    self.assertEqual(q[state][action], 0.0)

    def test_determinism(self):
        self.assertEqual(q_learning(steps=100), q_learning(steps=100))


def demo_dp():
    policy, values, history = policy_iteration()
    for index, (current, prediction) in enumerate(history):
        print(f"PI {index}: policy={current}, V(A)={prediction[0]:.6f}, V(B)={prediction[1]:.6f}")
    vi_policy, vi_values, sweeps, residual = value_iteration()
    print(f"VI: policy={vi_policy}, V(A)={vi_values[0]:.6f}, V(B)={vi_values[1]:.6f}, sweeps={sweeps}, residual={residual:.3e}")


def demo_control():
    q, policy, visits, reward_sum, resets = q_learning()
    print("Q-learning: Q=" + str({s: {a: round(v, 6) for a, v in row.items()} for s, row in q.items()}))
    values = evaluate_policy(policy)
    print(f"greedy policy={policy}, independently evaluated V(A)={values[0]:.6f}")
    print(f"training: steps=20000, visits={visits}, reward_sum={reward_sum:.1f}, true_terminal_resets={resets}")
    print("Training reward_sum is not the discounted greedy-policy value.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("dp", "control", "all", "test"), nargs="?", default="all")
    args = parser.parse_args()
    if args.mode == "test":
        unittest.main(argv=[__file__])
    else:
        if args.mode in ("dp", "all"):
            demo_dp()
        if args.mode in ("control", "all"):
            demo_control()
