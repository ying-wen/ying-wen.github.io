"""Off-policy Differential TD
行为策略各动作一半，目标策略因状态而异。动作重要性比同时乘bias与g更新，目标仍是pi的Poisson方程；无重放、无外部重置。
See docs/average-systems.md for equations, assumptions and diagnostics.
"""
import random
from pathlib import Path
try:
    from ._common import *
except ImportError:
    from _common import *

META = {
    "id": "differential_td_offpolicy",
    "name": "Off-policy Differential TD",
    "family": "average_systems",
    "implementation_kind": "teaching_method",
    "coverage_refs": [],
    "task": "average_prediction_offpolicy",
    "baseline": "wrong_behavior_mean_td",
    "metric": "参考状态对齐后的差分价值RMSE",
    "unit": "value",
    "higher_better": False,
    "budget": "environment_steps",
    "scope": "tabular-continuing-prediction",
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
    "description": "行为策略各动作一半，目标策略因状态而异。动作重要性比同时乘bias与g更新，目标仍是pi的Poisson方程；无重放、无外部重置。",
    "question": "怎样同时学到目标策略的奖励率与差分价值，而不把行为奖励率误当成目标？",
    "limitations": "平稳遍历有限MDP、表格表示、常数步长、1200步教学预算；不是神经网络稳定性或非平稳收敛证明。"
}

def update(h, rate, s, reward, sp, rho, alpha=.08, eta=.2):
    if rho < 0:
        raise ValueError('Importance ratio must be nonnegative')
    delta = reward-rate+h[sp]-h[s]
    new = h[:]
    new[s] += alpha*rho*delta
    return new, rate+eta*alpha*rho*delta, delta

def run(seed=0, steps=1200, emit=None):
    check_steps(steps)
    coverage(TARGET, BEHAVIOR)
    rng, rows, h, rate, s = random.Random(seed), [], [0., 0., 0.], 0., 0
    true_rate, true_bias = evaluate(TARGET)
    behavior_rate, _ = evaluate(BEHAVIOR)
    reward_total = 0.
    log(rows, 0, steps, emit, **prediction_diagnostics(h, rate, true_rate, true_bias),
        experienced_reward_rate=0., true_behavior_gain=behavior_rate, environment_updates=0)
    for t in range(1, steps+1):
        a = sample(BEHAVIOR[s], rng)
        reward, sp = transition(s, a, rng)
        rho = TARGET[s][a]/BEHAVIOR[s][a]
        h, rate, delta = update(h, rate, s, reward, sp, rho)
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

