"""MIT teaching implementations; Python 3.10+, standard library only.

Small exact examples isolate update semantics, not benchmark performance.
Run: python3 foundations_detail_lab.py all; python3 foundations_detail_lab.py test
"""
import argparse
import copy
import json
import math
import random
import unittest


# BEGIN prediction
def prediction(method="td", episodes=200, alpha=0.1):
    """A --0--> B --1--> terminal; gamma=.9, exact values [.9, 1]."""
    values = [0.0, 0.0, 0.0]
    trajectory = [(0, 0.0, 1, 0.9), (1, 1.0, 2, 0.0)]
    for _ in range(episodes):
        if method == "mc":
            ret = 0.0
            for state, reward, _, discount in reversed(trajectory):
                ret = reward + discount * ret
                values[state] += alpha * (ret - values[state])
        elif method == "td":
            for state, reward, nxt, discount in trajectory:
                delta = reward + discount * values[nxt] - values[state]
                values[state] += alpha * delta
        else:
            raise ValueError("Choose mc or td")
    return values[:2]
# END prediction


# BEGIN control
def control_target(reward, discount, next_q, method, next_action=None, probs=None):
    if discount == 0.0:
        return reward
    if method == "q":
        bootstrap = max(next_q)
    elif method == "sarsa":
        bootstrap = next_q[next_action]
    elif method == "expected":
        bootstrap = sum(p * q for p, q in zip(probs, next_q))
    else:
        raise ValueError(method)
    return reward + discount * bootstrap


def train_control(method, episodes=20000, seed=7):
    """Only one action at A; at B behavior chooses bad action with p=.1."""
    rng = random.Random(seed)
    q_a, q_b = 0.0, [0.0, 0.0]
    for _ in range(episodes):
        next_action = int(rng.random() < 0.1)
        target = control_target(0.0, 0.9, q_b, method, next_action, [0.9, 0.1])
        q_a += 0.01 * (target - q_a)
        reward = 1.0 if next_action == 0 else -1.0
        q_b[next_action] += 0.01 * (reward - q_b[next_action])
    return {"q_A": q_a, "q_B": q_b}
# END control


# BEGIN dqn
class TinyQ:
    """Two-layer ReLU Q-network: 2 state features -> 8 hidden -> 2 actions."""
    def __init__(self, rng, hidden=8):
        self.w1 = [[rng.uniform(-0.5, 0.5) for _ in range(2)] for _ in range(hidden)]
        self.b1 = [0.1] * hidden
        self.w2 = [[rng.uniform(-0.2, 0.2) for _ in range(hidden)] for _ in range(2)]
        self.b2 = [0.0, 0.0]

    def forward(self, state):
        # An integer state is a one-hot input; terminal state is never evaluated.
        hidden = [max(0.0, row[state] + b) for row, b in zip(self.w1, self.b1)]
        q = [sum(w * h for w, h in zip(row, hidden)) + b
             for row, b in zip(self.w2, self.b2)]
        return hidden, q

    def gradient(self, state, action, target):
        hidden, q = self.forward(state)
        error = q[action] - target  # gradient of 1/2 * squared error
        g = {"w1": [[0.0] * 2 for _ in hidden], "b1": [0.0] * len(hidden),
             "w2": [[0.0] * len(hidden) for _ in range(2)], "b2": [0.0, 0.0]}
        g["b2"][action] = error
        for j, h in enumerate(hidden):
            g["w2"][action][j] = error * h
            dh = error * self.w2[action][j] * (h > 0.0)
            g["w1"][j][state] = dh
            g["b1"][j] = dh
        return g

    def train_batch(self, frozen_samples, alpha):
        # All gradients use the SAME pre-update parameters.
        gradients = [self.gradient(s, a, y) for s, a, y in frozen_samples]
        for name in ("w1", "w2", "b1", "b2"):
            param = getattr(self, name)
            for i in range(len(param)):
                if isinstance(param[i], list):
                    for j in range(len(param[i])):
                        param[i][j] -= alpha * sum(g[name][i][j] for g in gradients) / len(gradients)
                else:
                    param[i] -= alpha * sum(g[name][i] for g in gradients) / len(gradients)


def dqn_target(online, target, reward, discount, next_state, double):
    if discount == 0.0:
        return reward  # do not even index a terminal observation
    _, target_q = target.forward(next_state)
    if double:
        _, online_q = online.forward(next_state)
        selected = max(range(2), key=lambda a: online_q[a])
        return reward + discount * target_q[selected]
    return reward + discount * max(target_q)


def train_dqn(steps=5000, seed=7, double=True):
    rng = random.Random(seed)
    online = TinyQ(rng)
    target = copy.deepcopy(online)
    replay, state = [], 0
    for t in range(steps):
        _, q = online.forward(state)
        action = rng.randrange(2) if rng.random() < 0.2 else max(range(2), key=lambda a: q[a])
        if state == 0 and action == 0:
            reward, discount, nxt = 0.0, 0.9, 1
        else:
            reward = 0.1 if state == 0 else (1.0 if action == 0 else -1.0)
            discount, nxt = 0.0, None
        replay.append((state, action, reward, discount, nxt))
        replay = replay[-256:]
        if len(replay) >= 16:
            batch = rng.sample(replay, 16)
            frozen = [(s, a, dqn_target(online, target, r, g, sp, double))
                      for s, a, r, g, sp in batch]
            online.train_batch(frozen, alpha=0.03)
        if (t + 1) % 25 == 0:
            target = copy.deepcopy(online)
        state = 0 if discount == 0.0 else nxt
    estimates = [online.forward(s)[1] for s in range(2)]
    reference = [[0.9, 0.1], [1.0, -1.0]]
    return {"Q": estimates, "reference": reference,
            "max_error": max(abs(estimates[s][a] - reference[s][a]) for s in range(2) for a in range(2))}
# END dqn


# BEGIN policy
def sigmoid(theta):
    if theta >= 0:
        return 1.0 / (1.0 + math.exp(-theta))
    exp_theta = math.exp(theta)
    return exp_theta / (1.0 + exp_theta)


def ppo_sample(theta, old_probability, action, advantage, epsilon=0.2):
    p = sigmoid(theta)
    probability = p if action == 1 else 1.0 - p
    ratio = probability / old_probability
    clipped = min(1.0 + epsilon, max(1.0 - epsilon, ratio))
    objective = min(ratio * advantage, clipped * advantage)
    inactive = (advantage > 0 and ratio > 1.0 + epsilon) or (advantage < 0 and ratio < 1.0 - epsilon)
    derivative = 0.0 if inactive else advantage * ratio * (action - p)
    return objective, derivative


def gae(rewards, values, next_values, discounts, trace_continues, lam=0.95):
    # At time-limit truncation, bootstrap may survive but the trace must not
    # jump into the next reset episode. next_values uses FINAL observation.
    out, tail = [0.0] * len(rewards), 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + discounts[t] * next_values[t] - values[t]
        tail = delta + discounts[t] * lam * trace_continues[t] * tail
        out[t] = tail
    return out


def train_ppo_bandit(rounds=100, seed=7):
    rng, theta = random.Random(seed), 0.0
    for _ in range(rounds):
        old_p = sigmoid(theta)
        batch = []
        for _ in range(64):
            action = int(rng.random() < old_p)
            # Exact old-policy baseline for this reward function.
            batch.append((old_p if action else 1.0 - old_p, action, action - old_p))
        for _ in range(4):
            grad = sum(ppo_sample(theta, *sample)[1] for sample in batch) / len(batch)
            theta += 0.1 * grad
    return {"probability_good_action": sigmoid(theta)}
# END policy


# BEGIN soft
def softmax(logits):
    shifted = [math.exp(x - max(logits)) for x in logits]
    return [x / sum(shifted) for x in shifted]


def soft_value(q, temperature):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    largest = max(q)
    return largest + temperature * math.log(sum(math.exp((x - largest) / temperature) for x in q))


def discrete_sac_target(reward, discount, probabilities, q1, q2, temperature):
    if discount == 0.0:
        return reward
    expectation = sum(p * (min(a, b) - temperature * math.log(p))
                      for p, a, b in zip(probabilities, q1, q2) if p > 0)
    return reward + discount * expectation


def categorical_actor(logits, q, temperature):
    probabilities = softmax(logits)
    top = max(logits)
    log_z = math.log(sum(math.exp(z - top) for z in logits))
    log_probs = [z - top - log_z for z in logits]
    cost = [temperature * logp - val for logp, val in zip(log_probs, q)]
    objective = sum(p * c for p, c in zip(probabilities, cost))
    # Gradient of sum pi(a) [temperature log pi(a) - Q(a)].
    gradient = [p * (c - objective) for p, c in zip(probabilities, cost)]
    return objective, gradient


def train_soft_bandit(temperature=0.5, iterations=2000):
    q, logits = [0.0, 1.0], [0.0, 0.0]
    for _ in range(iterations):
        _, grad = categorical_actor(logits, q, temperature)
        logits = [z - 0.1 * g for z, g in zip(logits, grad)]
    return {"learned": softmax(logits), "exact": softmax([v / temperature for v in q]),
            "soft_value": soft_value(q, temperature)}
# END soft


# BEGIN dyna
def q_backup(q, s, a, reward, discount, nxt, alpha):
    bootstrap = max(q[nxt]) if discount else 0.0
    q[s][a] += alpha * (reward + discount * bootstrap - q[s][a])


def dyna_chain(planning=5, steps=1000, seed=7):
    """Deterministic 5-state chain. Right at 4 gives 1 and truly terminates."""
    rng, q, model, state = random.Random(seed), [[0.0, 0.0] for _ in range(5)], {}, 0
    for _ in range(steps):
        greedy = [a for a in range(2) if q[state][a] == max(q[state])]
        action = rng.randrange(2) if rng.random() < 0.2 else rng.choice(greedy)
        nxt = max(0, state - 1) if action == 0 else state + 1
        reward, discount = (1.0, 0.0) if nxt == 5 else (0.0, 0.9)
        q_backup(q, state, action, reward, discount, nxt, 0.1)
        model[(state, action)] = (reward, discount, nxt)
        for _ in range(planning):
            simulated_s, simulated_a = rng.choice(list(model))
            r, g, sp = model[(simulated_s, simulated_a)]
            q_backup(q, simulated_s, simulated_a, r, g, sp, 0.1)
        state = 0 if not discount else nxt
    return {"Q": q, "right_action_reference": [0.9 ** (4 - s) for s in range(5)],
            "real_steps": steps, "planning_updates": planning * steps}
# END dyna


class Checks(unittest.TestCase):
    def test_prediction_reference(self):
        for method in ["td", "mc"]:
            for pred, ref in zip(prediction(method), [0.9, 1.0]):
                self.assertAlmostEqual(pred, ref, places=6)

    def test_first_episode(self):
        self.assertEqual(prediction("td", 1), [0.0, 0.1])
        self.assertAlmostEqual(prediction("mc", 1)[0], 0.09)

    def test_control_targets(self):
        self.assertEqual(control_target(0, 0.9, [1, -1], "q"), 0.9)
        self.assertEqual(control_target(0, 0.9, [1, -1], "sarsa", 1), -0.9)
        self.assertAlmostEqual(control_target(0, 0.9, [1, -1], "expected", probs=[0.9, 0.1]), 0.72)

    def test_control_terminal_without_next_state(self):
        for method in ["q", "sarsa", "expected"]:
            self.assertEqual(control_target(1, 0, None, method), 1)

    def test_network_gradient(self):
        net = TinyQ(random.Random(8))
        gradient = net.gradient(0, 1, 0.7)
        for name in ["w1", "w2", "b1", "b2"]:
            param = getattr(net, name)
            for i in range(len(param)):
                for j in (range(len(param[i])) if isinstance(param[i], list) else [None]):
                    row, k = (param[i], j) if j is not None else (param, i)
                    old, eps = row[k], 1e-6
                    row[k] = old + eps
                    plus = 0.5 * (net.forward(0)[1][1] - 0.7) ** 2
                    row[k] = old - eps
                    minus = 0.5 * (net.forward(0)[1][1] - 0.7) ** 2
                    row[k] = old
                    analytic = gradient[name][i][j] if j is not None else gradient[name][i]
                    self.assertAlmostEqual((plus - minus) / (2 * eps), analytic, places=7)

    def test_double_selection(self):
        online, target = TinyQ(random.Random(1)), TinyQ(random.Random(2))
        online.w2 = [[0.0] * 8, [0.0] * 8]
        target.w2 = [[0.0] * 8, [0.0] * 8]
        online.b2, target.b2 = [3, 2], [1, 4]
        self.assertEqual(dqn_target(online, target, 0, 0.9, 0, True), 0.9)
        self.assertEqual(dqn_target(online, target, 0, 0.9, 0, False), 3.6)
        self.assertEqual(dqn_target(online, target, 2, 0, None, True), 2)

    def test_dqn_reference(self):
        self.assertLess(train_dqn()["max_error"], 0.1)

    def test_ppo_clipping(self):
        self.assertEqual(ppo_sample(math.log(3), 0.5, 1, 1)[1], 0)
        self.assertLess(ppo_sample(math.log(3), 0.5, 1, -1)[1], 0)

    def test_sigmoid_extremes(self):
        self.assertEqual(sigmoid(-1000), 0.0)
        self.assertEqual(sigmoid(1000), 1.0)

    def test_ppo_gradient(self):
        for theta, action, advantage in [(0.1, 1, 2), (-0.1, 0, -1), (2, 1, 1)]:
            eps = 1e-6
            finite = (ppo_sample(theta+eps, 0.5, action, advantage)[0] - ppo_sample(theta-eps, 0.5, action, advantage)[0]) / (2*eps)
            self.assertAlmostEqual(finite, ppo_sample(theta, 0.5, action, advantage)[1], places=7)

    def test_gae_truncation(self):
        out = gae([1, 10], [0, 0], [2, 0], [.9, 0], [0, 0])
        self.assertEqual(out, [2.8, 10])

    def test_actor_gradient(self):
        logits, q, temperature, eps = [0.1, -0.3], [0, 1], 0.5, 1e-6
        _, grad = categorical_actor(logits, q, temperature)
        for i in range(2):
            hi, lo = logits[:], logits[:]
            hi[i] += eps
            lo[i] -= eps
            finite = (categorical_actor(hi, q, temperature)[0] - categorical_actor(lo, q, temperature)[0]) / (2 * eps)
            self.assertAlmostEqual(finite, grad[i], places=7)

    def test_soft_solution(self):
        out = train_soft_bandit()
        self.assertAlmostEqual(out["learned"][1], out["exact"][1], places=5)
        self.assertAlmostEqual(soft_value([3, 3], .5), 3 + .5 * math.log(2))

    def test_soft_terminal_and_extreme_logits(self):
        self.assertEqual(discrete_sac_target(2, 0, None, None, None, .5), 2)
        objective, gradient = categorical_actor([1000, -1000], [0, 1], .5)
        self.assertTrue(math.isfinite(objective))
        self.assertTrue(all(math.isfinite(g) for g in gradient))

    def test_dyna_reference(self):
        out = dyna_chain()
        for row, ref in zip(out["Q"], out["right_action_reference"]):
            self.assertAlmostEqual(row[1], ref, places=3)


def run(name):
    return {"value": lambda: {m: prediction(m) for m in ["mc", "td"]},
            "control": lambda: {m: train_control(m) for m in ["q", "sarsa", "expected"]},
            "deep-value": train_dqn, "policy": train_ppo_bandit,
            "soft-control": train_soft_bandit, "dyna": dyna_chain}[name]()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["all", "test", "value", "control", "deep-value", "policy", "soft-control", "dyna"])
    args = parser.parse_args()
    if args.mode == "test":
        unittest.main(argv=[__file__], verbosity=2)
    else:
        result = ({m: run(m) for m in ["value", "control", "deep-value", "policy", "soft-control", "dyna"]}
                  if args.mode == "all" else run(args.mode))
        print(json.dumps(result, indent=2))
