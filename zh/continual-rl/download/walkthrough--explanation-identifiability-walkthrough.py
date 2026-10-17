"""Read a controlled behavioral-identification example without sampling data.

Run: python3 explanation-identifiability-walkthrough.py
Only standard libraries. Fraction-based closed forms and direct binomial
likelihoods independently check the JavaScript recurrence/score-root kernel.
The printed recovery matrix integrates all possible counts of two fixed rules;
it is neither human data nor a full recovery study of fitted model families.
"""
from fractions import Fraction
from math import comb, exp, log, isclose
import json


def sigmoid(z):
    return 1 / (1 + exp(-z))


def probability(alpha, beta, n):
    # Closed form, independent of the companion's step-by-step update.
    return sigmoid(float(beta * (1 - (1 - alpha) ** n)))


def inverse(p1, p2):
    z1, z2 = log(p1 / (1 - p1)), log(p2 / (1 - p2))
    alpha = 2 - z2 / z1
    if not (0 < alpha <= 1):
        raise ValueError('empirical frequencies are outside the model')
    return dict(alpha=alpha, beta=z1 / alpha)


def likelihood(alpha, beta, rows):
    # Direct Bernoulli log probabilities, independent of JS softplus.
    return sum(k * log(p) + (n - k) * log(1 - p)
               for feedback, n, k in rows
               for p in [probability(alpha, beta, feedback)])


def profile(alpha, rows):
    # Golden-section objective search rather than the JS score-root search.
    low, high, ratio = 0.0, 100 / alpha, (5 ** .5 - 1) / 2
    for _ in range(150):
        left, right = high - ratio * (high - low), low + ratio * (high - low)
        # Stable log probabilities avoid saturation in this broad search.
        def stable_ll(beta):
            total = 0.0
            for feedback, n, k in rows:
                z = beta * float(1 - (1 - alpha) ** feedback)
                total -= k * log(1 + exp(-z)) + (n - k) * (z + log(1 + exp(-z)))
            return total
        if stable_ll(left) < stable_ll(right):
            low = left
        else:
            high = right
    beta = (low + high) / 2
    return dict(alpha=alpha, beta=beta, logLikelihood=likelihood(alpha, beta, rows))


def recovery(total=20, threshold=2):
    first = sigmoid(1)
    matrix = []
    for second in [sigmoid(1.8), first]:
        mass = [0.0, 0.0]
        for k1 in range(total + 1):
            for k2 in range(total + 1):
                weight = (comb(total, k1) * first ** k1 * (1 - first) ** (total - k1)
                          * comb(total, k2) * second ** k2 * (1 - second) ** (total - k2))
                mass[0 if k2 - k1 >= threshold else 1] += weight
        matrix.append(mass)
    return matrix


def walkthrough():
    rows = [(1, 100, 73), (2, 100, 86)]
    recovered = inverse(.73, .86)
    examples = [(Fraction(1, 5), 5), (Fraction(1, 2), 2)]
    for alpha, beta in examples:
        assert isclose(probability(alpha, beta, 1), sigmoid(1), abs_tol=1e-14)
        oracle = inverse(probability(alpha, beta, 1), probability(alpha, beta, 2))
        assert isclose(oracle['alpha'], float(alpha), abs_tol=1e-13)
        assert isclose(oracle['beta'], beta, abs_tol=1e-12)
    matrix = recovery()
    assert all(isclose(sum(row), 1, abs_tol=1e-12) for row in matrix)
    return dict(participantCount=0, sampledRuns=0, countOrigin='fixed illustrative counts',
                oracleRecovery='algebraic round-trip only; not noisy parameter recovery', fitted=recovered,
                candidateLikelihoods=[likelihood(alpha, beta, rows) for alpha, beta in examples],
                profile=[profile(alpha, rows) for alpha in [.1, .2, .5, 1]],
                fixedGeneratorRecovery=matrix)


if __name__ == '__main__':
    print(json.dumps(walkthrough(), ensure_ascii=False, indent=2))
