"""Exact arithmetic for A -> B -> A -> terminal, and learning-rate credit.

Run: python3 credit_meta_walkthrough.py
No random training. Python uses Fraction, direct n-step target sums and forward
sensitivity; the JavaScript companion uses recursive targets/reverse adjoints.
"""
from fractions import Fraction as F
import json

X = [[F(1), F(0)], [F(1), F(1)], [F(1), F(0)], [F(0), F(0)]]
R = [F(1), F(0), F(2)]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def td(alpha=F(1, 2), lam=F(1, 2), gamma=F(1), true_online=False):
    w, z, old_prediction = [F(0), F(0)], [F(0), F(0)], F(0)
    history, rows = [w[:]], []
    for t, reward in enumerate(R):
        value, following = dot(w, X[t]), dot(w, X[t + 1])
        delta, gap = reward + gamma * following - value, value - old_prediction
        overlap = dot(z, X[t])
        z = [gamma * lam * zi + (1 - alpha * gamma * lam * overlap if true_online else 1) * xi
             for zi, xi in zip(z, X[t])]
        inc = [alpha * ((delta + gap) * zi - gap * xi) if true_online else alpha * delta * zi
               for zi, xi in zip(z, X[t])]
        w = [wi + di for wi, di in zip(w, inc)]
        rows.append(dict(t=t, delta=delta, trace=z[:], prediction=value,
                         nextPrediction=following, oldPrediction=old_prediction,
                         versionGap=gap, increment=inc, weight=w[:]))
        history.append(w[:])
        old_prediction = following
    return dict(history=history, rows=rows)


def forward(alpha=F(1, 2), lam=F(1, 2), gamma=F(1)):
    history, targets = [[F(0), F(0)]], []
    for horizon in range(1, len(R) + 1):
        helper, row = [F(0), F(0)], []
        for t in range(horizon):
            target = F(0)
            for n in range(1, horizon - t + 1):
                weight = lam ** (n - 1) * (1 - lam if n < horizon - t else 1)
                sampled = sum(gamma ** k * R[t + k] for k in range(n))
                bootstrap = gamma ** n * dot(history[t + n - 1], X[t + n])
                target += weight * (sampled + bootstrap)
            error = target - dot(helper, X[t])
            helper = [w + alpha * error * x for w, x in zip(helper, X[t])]
            row.append(target)
        history.append(helper)
        targets.append(row)
    return dict(history=history, targets=targets)


def meta(alpha=F(1, 2)):
    run = td(alpha)
    paths = [[F(0), F(0)] for _ in R]
    stopped = [F(0), F(0)]
    for t, row in enumerate(run['rows']):
        z, delta = row['trace'], row['delta']
        direction = [b - a for a, b in zip(X[t], X[t + 1])]
        # Each column varies one alpha_t; summing columns ties all alphas.
        for k in range(len(R)):
            propagated = dot(direction, paths[k])
            paths[k] = [h + alpha * zi * propagated + (delta * zi if k == t else 0)
                        for h, zi in zip(paths[k], z)]
        stopped_error = -dot(X[t], stopped)
        stopped = [h + delta * zi + alpha * zi * stopped_error for h, zi in zip(stopped, z)]
    h = [sum(path[j] for path in paths) for j in range(2)]
    prediction = sum(run['history'][-1])
    residual = prediction - 2
    prediction_paths = [sum(path) for path in paths]
    gradient_paths = [residual * v for v in prediction_paths]
    reused_residual = run['history'][-1][0] - 2
    return dict(evaluationFeature=[1, 1], target=2, prediction=prediction,
                residual=residual, loss=residual**2 / 2, sensitivity=h,
                full=sum(gradient_paths), direct=gradient_paths[-1],
                stopBootstrap=residual * sum(stopped),
                perStepPredictionDerivative=prediction_paths, perStepGradient=gradient_paths,
                reusedTrainLoss=reused_residual**2 / 2, reusedTrainGradient=reused_residual * h[0])


def as_numbers(value):
    if isinstance(value, F):
        return float(value)
    if isinstance(value, dict):
        return {k: as_numbers(v) for k, v in value.items()}
    if isinstance(value, list):
        return [as_numbers(v) for v in value]
    return value


def checks():
    assert td()['history'][-1] == [F(29, 16), F(3, 8)]
    for lam in (F(0), F(1, 2), F(1)):
        assert forward(lam=lam)['history'] == td(lam=lam, true_online=True)['history']
    assert meta()['sensitivity'] == [F(11, 4), F(1, 2)]
    assert meta()['full'] == F(39, 64)
    assert meta()['direct'] == F(81, 128)
    assert meta()['stopBootstrap'] == F(273, 512)
    eps = F(1, 100000)
    difference = (meta(F(1, 2) + eps)['loss'] - meta(F(1, 2) - eps)['loss']) / (2 * eps)
    assert abs(difference - meta()['full']) < F(1, 10000000)


if __name__ == '__main__':
    checks()
    print(json.dumps(as_numbers(dict(traditional=td(), trueOnline=td(true_online=True),
                                   forward=forward(), meta=meta())), indent=2))
