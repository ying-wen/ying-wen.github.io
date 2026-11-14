"""Three-state continuing MDP and exact *evaluation-only* numerical oracles.

All policies are irreducible and aperiodic: each transition has probability
0.2/3 of each destination. The learner is never given P or exact solutions,
except in the explicitly known-model planning experiments.
"""
import itertools
import math

REWARDS = [[.1, -.05], [.5, .05], [.2, 1.2]]
DESTINATIONS = [[1, 2], [0, 2], [0, 1]]
P = [[[.2/3 + (.8 if sp == DESTINATIONS[s][a] else 0.)
       for sp in range(3)] for a in range(2)] for s in range(3)]
TARGET = [[.15, .85], [.25, .75], [.2, .8]]
BEHAVIOR = [[.5, .5] for _ in range(3)]


def check_steps(steps):
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError('steps must be a positive integer')


def sample(probabilities, rng):
    u, total = rng.random(), 0.
    for i, probability in enumerate(probabilities):
        total += probability
        if u < total:
            return i
    return len(probabilities)-1


def transition(s, a, rng):
    # No terminal flag: the data stream never ends or resets at a log boundary.
    return REWARDS[s][a], sample(P[s][a], rng)


def coverage(target, behavior):
    for pi, b in zip(target, behavior):
        if any(p > 0 and q <= 0 for p, q in zip(pi, b)):
            raise ValueError('Target actions require behavior support')


def solve(matrix, rhs):
    """Partial-pivot Gaussian elimination, for tiny exact-policy evaluations."""
    n = len(rhs)
    aug = [list(map(float, row))+[float(y)] for row, y in zip(matrix, rhs)]
    for j in range(n):
        k = max(range(j, n), key=lambda i: abs(aug[i][j]))
        if abs(aug[k][j]) < 1e-12:
            raise ValueError('Singular system: check recurrent classes/gauge')
        aug[j], aug[k] = aug[k], aug[j]
        divisor = aug[j][j]
        aug[j] = [x/divisor for x in aug[j]]
        for i in range(n):
            if i != j:
                factor = aug[i][j]
                aug[i] = [x-factor*y for x, y in zip(aug[i], aug[j])]
    return [row[-1] for row in aug]


def evaluate(policy, transitions=P, rewards=REWARDS):
    """Solve g + h(s) - P_pi h(s) = r_pi(s), with h(0)=0.

    Unknowns are [g, h(0), ..., h(n-1)]. This uses the *true* model only to
    score a frozen policy. It never supplies a learning target or action.
    """
    n = len(policy)
    rows, rhs = [], []
    for s in range(n):
        row = [1.]+[float(j == s)-sum(policy[s][a]*transitions[s][a][j]
                    for a in range(len(policy[s]))) for j in range(n)]
        rows.append(row)
        rhs.append(sum(policy[s][a]*rewards[s][a] for a in range(len(policy[s]))))
    rows.append([0., 1.]+[0.]*(n-1))
    rhs.append(0.)
    solution = solve(rows, rhs)
    return solution[0], solution[1:]


def greedy(q):
    return [[float(a == max(range(2), key=lambda b: q[s][b])) for a in range(2)]
            for s in range(3)]


def optimal():
    # Finite enumeration is an evaluation oracle, never part of learned control.
    candidates = []
    for actions in itertools.product(range(2), repeat=3):
        pi = [[float(a == actions[s]) for a in range(2)] for s in range(3)]
        g, h = evaluate(pi)
        candidates.append((g, h, pi))
    return max(candidates, key=lambda x: x[0])


def prediction_diagnostics(h, rate, true_rate, true_bias):
    anchored = [v-h[0] for v in h]
    error = math.sqrt(sum((x-y)**2 for x, y in zip(anchored, true_bias))/len(h))
    return {'value': error, 'gain': rate, 'gain_abs_error': abs(rate-true_rate),
            'true_gain': true_rate, 'bias_0': anchored[0],
            'bias_1': anchored[1], 'bias_2': anchored[2], 'raw_bias_offset': h[0]}


def log(rows, t, steps, emit, **diagnostics):
    if t == 0 or t == steps or t % max(1, steps//20) == 0:
        row = {'step': t, 'phase': 'evaluate' if t else 'initialize', **diagnostics}
        rows.append(row)
        if emit:
            emit(row)


def control_diagnostics(q, rate, reward_total, t, model_backups=0):
    actual_gain, _ = evaluate(greedy(q))
    best_gain, _, _ = optimal()
    residual = max(abs(REWARDS[s][a]-rate + sum(P[s][a][sp]*max(q[sp])
                    for sp in range(3))-q[s][a]) for s in range(3) for a in range(2))
    return {'value': actual_gain, 'optimal_gain': best_gain,
            'optimality_gap': best_gain-actual_gain,
            'gain_estimate': rate, 'gain_estimate_abs_error': abs(rate-best_gain),
            'experienced_reward_rate': reward_total/t if t else 0.,
            'environment_updates': t, 'model_backups': model_backups,
            'total_backups': t+model_backups, 'bellman_residual': residual}
