#!/usr/bin/env python3
"""Four small CRL teaching diagnostics; Python 3 stdlib only, no paper-reproduction claim.

Outputs per-seed, per-window CSV. Rewards/targets are generated locally; no downloads.
IDBD uses the linear diagonal meta-gradient approximation (Sutton, 1992).
Clipping beta is an explicitly added numerical guard, not a NetworkIDBD implementation.
"""
import argparse
import csv
import math
import random
from pathlib import Path


def bandit(seed=0, steps=4000, alpha=0.1, epsilon=0.1, switch=None, **_):
    switch = steps // 2 if switch is None else switch
    rng = random.Random(seed)
    # Common exogenous uniforms across methods; outcomes still depend on each action.
    data = [(rng.random(), rng.random(), rng.random(), rng.random()) for _ in range(steps)]
    rows = []
    for method in ('sample_mean', 'constant_step'):
        q, n, rewards, choices = [0., 0.], [0, 0], [], []
        for t, (explore, choose, tie, reward_u) in enumerate(data):
            action = int(choose >= 0.5) if explore < epsilon else (int(tie >= 0.5) if q[0] == q[1] else int(q[1] > q[0]))
            probs = (0.8, 0.2) if t < switch else (0.2, 0.8)
            r = float(reward_u < probs[action])
            n[action] += 1
            rate = 1 / n[action] if method == 'sample_mean' else alpha
            q[action] += rate * (r - q[action])
            rewards.append(r)
            choices.append(action)
            # End a window at the switch too; never mix regimes in a summary.
            if (t + 1) % 100 == 0 or t + 1 == switch or t == steps - 1:
                metrics = {'window_reward':sum(rewards)/len(rewards),
                           'choice_b':sum(choices)/len(choices), 'q_a':q[0], 'q_b':q[1]}
                for metric, value in metrics.items():
                    rows.append(dict(lab='bandit', seed=seed, method=method, step=t+1,
                                     metric=metric, value=value, window_steps=len(rewards)))
                rewards, choices = [], []
    return rows


def memory(seed=0, steps=4000, **_):
    # Balanced hidden cues make the information limit exact, not a lucky seed.
    # Here `steps` counts trials; each trial contains a cue, 10 blanks, a choice.
    if steps < 2 or steps % 2:
        raise ValueError('Memory requires a positive even number of trials (at least 2).')
    cues = [i % 2 for i in range(steps)]
    random.Random(seed).shuffle(cues)
    correct = {'no_memory_fixed_left':0, 'one_bit_oracle':0}
    for cue in cues:
        state = None
        for observation in [cue] + [None] * 10:
            if observation is not None:
                state = observation
        correct['no_memory_fixed_left'] += int(0 == cue)
        correct['one_bit_oracle'] += int(state == cue)
    return [dict(lab='memory', seed=seed, method=method, step=steps, metric='accuracy', value=count/steps)
            for method, count in correct.items()]


def credit(seed=0, steps=8000, dims=16, noise=1., meta=0.01, **_):
    rng = random.Random(seed)
    weights = {'sgd': [0.] * dims, 'idbd': [0.] * dims}
    beta, h = [math.log(0.01)] * dims, [0.] * dims
    errors = {'sgd': [], 'idbd': []}
    rows = []
    for t in range(steps):
        x = [float(rng.random() < 0.1) for _ in range(dims)]
        signal = x[0]
        target = signal + rng.gauss(0, noise)
        for method in weights:
            w = weights[method]
            pred = sum(wi * xi for wi, xi in zip(w, x))
            delta = target - pred
            # Evaluate against the predictable component before observing target.
            errors[method].append((pred - signal) ** 2)
            for i, xi in enumerate(x):
                if method == 'idbd':
                    beta[i] = max(-12., min(-1., beta[i] + meta * delta * xi * h[i]))
                    rate = math.exp(beta[i])
                    w[i] += rate * delta * xi
                    h[i] = h[i] * max(0., 1. - rate * xi * xi) + rate * delta * xi
                else:
                    w[i] += 0.01 * delta * xi
        if (t + 1) % 100 == 0 or t == steps - 1:
            for method, values in errors.items():
                rows.append(dict(lab='credit', seed=seed, method=method, step=t+1, metric='signal_mse', value=sum(values)/len(values)))
                errors[method] = []
            rows.append(dict(lab='credit', seed=seed, method='idbd', step=t+1, metric='signal_alpha', value=math.exp(beta[0])))
            rows.append(dict(lab='credit', seed=seed, method='idbd', step=t+1, metric='noise_alpha', value=sum(math.exp(b) for b in beta[1:])/(dims-1)))
    return rows


def retention(seed=0, steps=1000, **_):
    # A deterministic counterexample: forgetting can occur without a decline in trainability.
    alpha, w, rows = 0.05, 0., []
    for _ in range(steps):
        w += alpha * (1. - w)
    fresh = 0.
    for t in range(steps):
        w += alpha * (-1. - w)
        fresh += alpha * (-1. - fresh)
        if (t+1) % 20 == 0 or t == steps-1:
            for name, value in [('old_task_error', (1-w)**2), ('new_task_error_old', (-1-w)**2), ('new_task_error_fresh', (-1-fresh)**2)]:
                rows.append(dict(lab='retention',seed=seed,method='linear',step=t+1,metric=name,value=value))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('lab', choices=['bandit','memory','credit','retention'])
    p.add_argument('--seeds', type=int, help='Default: 1 for memory/retention; 5 otherwise.')
    p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--alpha', type=float, default=0.1)
    p.add_argument('--epsilon', type=float, default=0.1)
    p.add_argument('--switch', type=int, default=None, help='Bandit: switch AFTER this many steps; default halfway.')
    p.add_argument('--dims', type=int, default=16)
    p.add_argument('--noise', type=float, default=1.)
    p.add_argument('--meta', type=float, default=0.01)
    p.add_argument('--out', type=Path, default=Path('crl-results.csv'))
    a = p.parse_args()
    if a.seeds is None:
        a.seeds = 1 if a.lab in ('memory', 'retention') else 5
    if not all(math.isfinite(v) for v in (a.alpha, a.epsilon, a.noise, a.meta)):
        p.error('Parameters must be finite.')
    if a.seeds < 1 or a.steps < 2 or a.dims < 2 or not 0 < a.alpha <= 1 or not 0 <= a.epsilon <= 1 or a.noise < 0 or a.meta < 0:
        p.error('Invalid experiment parameters.')
    if a.lab == 'bandit' and a.switch is not None and not 0 < a.switch < a.steps:
        p.error('--switch must lie strictly between 0 and --steps.')
    if a.lab != 'bandit' and a.switch is not None:
        p.error('--switch is only used by bandit.')
    if a.lab == 'memory' and a.steps % 2:
        p.error('Memory requires an even --steps count for balanced cue trials.')
    if a.lab == 'retention' and a.seeds != 1:
        p.error('Retention is deterministic; use --seeds 1, not repeated identical evidence.')
    if a.out.exists():
        p.error(f'{a.out} already exists. Choose a new output name to preserve previous results.')
    if not a.out.parent.is_dir():
        p.error('Output parent directory does not exist; create it first.')
    rows = [r for seed in range(a.seeds) for r in globals()[a.lab](seed=seed, **vars(a))]
    if not all(math.isfinite(r['value']) for r in rows):
        raise ValueError('Non-finite result; inspect numerical stability.')
    for row in rows:
        row.update(total_steps=a.steps, alpha=a.alpha, epsilon=a.epsilon,
                   switch_step=(a.steps//2 if a.switch is None else a.switch) if a.lab=='bandit' else '',
                   dims=a.dims, noise=a.noise, meta=a.meta)
    with a.out.open('x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['lab','seed','method','step','metric','value',
                                             'window_steps','total_steps','alpha','epsilon','switch_step','dims','noise','meta'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'{a.lab}: {a.seeds} seeds, {a.steps} steps, {len(rows)} rows -> {a.out}')
    print('Teaching diagnostic only. Inspect all seeds; do not infer a general paper result.')


if __name__ == '__main__':
    main()
