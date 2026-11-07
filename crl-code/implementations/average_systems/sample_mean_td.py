"""On-policy reward-mean TD
同策略的真实奖励均值估计g，表格TD估计bias。它是正确的on-policy对照，不应被误称为off-policy方法。逐样本均值不适合直接追踪突然漂移。
See docs/average-systems.md for equations, assumptions and diagnostics.
"""
import random
from pathlib import Path
try:
    from ._common import *
except ImportError:
    from _common import *

META = {
    "id": "sample_mean_td",
    "name": "On-policy reward-mean TD",
    "family": "average_systems",
    "implementation_kind": "component_experiment",
    "coverage_refs": [],
    "task": "average_prediction_onpolicy",
    "baseline": "differential_td_onpolicy",
    "metric": "参考状态对齐后的差分价值RMSE",
    "unit": "value",
    "higher_better": False,
    "budget": "environment_steps",
    "scope": "controlled-rate-estimator-experiment",
    "chapter_paths": [
        "algorithms/average-reward/"
    ],
    "sources": [
        "https://proceedings.mlr.press/v139/wan21a.html",
        "https://github.com/abhisheknaik96/average-reward-methods"
    ],
    "defaults": {
        "steps": 1200
    },
    "description": "同策略的真实奖励均值估计g，表格TD估计bias。它是正确的on-policy对照，不应被误称为off-policy方法。逐样本均值不适合直接追踪突然漂移。",
    "question": "怎样同时学到目标策略的奖励率与差分价值，而不把行为奖励率误当成目标？",
    "limitations": "平稳遍历有限MDP、表格表示、常数步长、1200步教学预算；不是神经网络稳定性或非平稳收敛证明。"
}

def update(h, rate, s, reward, sp, count, alpha=.08):
    delta = reward-rate+h[sp]-h[s]
    new = h[:]
    new[s] += alpha*delta
    # The unweighted mean is valid here ONLY because b = pi.
    return new, rate+(reward-rate)/count, delta

def run(seed=0, steps=1200, emit=None):
    check_steps(steps)
    coverage(TARGET, TARGET)
    rng, rows, h, rate, s = random.Random(seed), [], [0., 0., 0.], 0., 0
    true_rate, true_bias = evaluate(TARGET)
    behavior_rate, _ = evaluate(TARGET)
    reward_total = 0.
    log(rows, 0, steps, emit, **prediction_diagnostics(h, rate, true_rate, true_bias),
        experienced_reward_rate=0., true_behavior_gain=behavior_rate, environment_updates=0)
    for t in range(1, steps+1):
        a = sample(TARGET[s], rng)
        reward, sp = transition(s, a, rng)
        h, rate, delta = update(h, rate, s, reward, sp, t)
        reward_total += reward
        s = sp
        log(rows, t, steps, emit, **prediction_diagnostics(h, rate, true_rate, true_bias),
            experienced_reward_rate=reward_total/t, true_behavior_gain=behavior_rate,
            environment_updates=t)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)

