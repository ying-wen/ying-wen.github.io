"""Problem definitions and logging only. Every algorithm has its own update loop."""
import math
import random


def validate(steps):
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")


def log(rows, step, value, steps, emit, phase="learning", **diagnostics):
    if step == 0 or step == steps or step % max(1, steps // 24) == 0:
        row = {"step": step, "value": value, "phase": phase, **diagnostics}
        rows.append(row)
        if emit:
            emit(row)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def baird_features():
    """Seven states, eight fixed features. Each feature vector has unit norm.

    Under behavior b: solid action with probability 1/7 leads to lower state;
    dashed action with probability 6/7 leads uniformly to six upper states.
    Thus d_b is uniform. Target pi always uses solid; all rewards are zero.
    Enumerating the importance-weighted expectation removes sampling noise,
    not the off-policy state-distribution mismatch.
    """
    result = []
    for s in range(7):
        x = [0.0] * 8
        if s < 6:
            x[s], x[7] = 2.0, 1.0
        else:
            x[6], x[7] = 1.0, 2.0
        result.append([v / math.sqrt(5.0) for v in x])
    return result


BAIRD_X = baird_features()
BAIRD_GAMMA = 0.99


def baird_init(seed):
    rng = random.Random(seed)
    w = [1.0 + rng.uniform(-0.01, 0.01) for _ in range(8)]
    w[6] = 10.0
    return w


def baird_error(w):
    # The target value is exactly zero. Log scale reveals growth without clipping.
    rmse = math.sqrt(sum(dot(x, w) ** 2 for x in BAIRD_X) / 7)
    return math.log10(1.0 + rmse), rmse


def value(w, x):
    """Four-parameter smooth shared approximator a*tanh(b*x+c)+d."""
    a, b, c, d = w
    return a * math.tanh(b * x + c) + d


def gradient(w, x):
    a, b, c, _ = w
    h = math.tanh(b * x + c)
    return [h, a * (1.0 - h * h) * x, a * (1.0 - h * h), 1.0]


def nonlinear_init(seed):
    rng = random.Random(seed)
    return [0.3 + rng.uniform(-0.03, 0.03), 0.4, 0.0, 0.0]


def regression_context(step, steps):
    # Context availability changes. The two conditional target functions do not.
    if step <= steps // 3:
        return 0, "context_A"
    if step <= 2 * steps // 3:
        return 1, "context_B"
    return 0, "return_A"


def regression_metrics(parameters, separate=False):
    predictions = [value(parameters[s] if separate else parameters, float(s + 1)) for s in range(2)]
    errors = [predictions[0] - 0.5, predictions[1] + 0.5]
    return math.sqrt(sum(e * e for e in errors) / 2), errors


def two_state_error(w):
    # A->B reward 0, B->A reward 1, gamma=.8, so v_A=20/9, v_B=25/9.
    return math.sqrt(((value(w, -1.0) - 20 / 9) ** 2 + (value(w, 1.0) - 25 / 9) ** 2) / 2)
