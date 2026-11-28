"""Differential Q · multistate control
三状态双动作遍历MDP，epsilon=0.25持续探索；同时学最优奖励率与动作差分价值。评估冻结贪心策略的真实奖励率，不把探索轨迹均值当目标。
"""
import random
from pathlib import Path
try:
    from ._common import *
except ImportError:
    from _common import *

META = {
    "id": "differential_q_multistate",
    "name": "Differential Q · multistate control",
    "family": "average_systems",
    "implementation_kind": "teaching_method",
    "coverage_refs": [],
    "task": "average_control_learned_model",
    "baseline": "differential_dyna",
    "metric": "冻结贪心策略的精确平均奖励",
    "unit": "reward_rate",
    "higher_better": True,
    "budget": "environment_steps",
    "scope": "tabular-continuing-control",
    "chapter_paths": [
        "algorithms/average-reward/",
        "construction/planning/"
    ],
    "sources": [
        "https://proceedings.mlr.press/v139/wan21a.html",
        "https://github.com/abhisheknaik96/average-reward-methods"
    ],
    "defaults": {
        "steps": 1200
    },
    "description": "三状态双动作遍历MDP，epsilon=0.25持续探索；同时学最优奖励率与动作差分价值。评估冻结贪心策略的真实奖励率，不把探索轨迹均值当目标。",
    "question": "Q与奖励率的联合更新怎样改变多状态中的长期控制？",
    "limitations": "平稳有限表格MDP与常数步长教学实验；不声称在1200步收敛，也不外推神经网络保证。"
}


def update(q, rate, s, a, reward, sp, alpha=.08, eta=.1):
    # Off-policy control: conditional (s,a) dynamics already correct; no pi/b.
    delta = reward-rate+max(q[sp])-q[s][a]
    new = [row[:] for row in q]
    new[s][a] += alpha*delta
    return new, rate+eta*alpha*delta, delta


def run(seed=0, steps=1200, emit=None):
    check_steps(steps)
    rng, rows = random.Random(seed), []
    q, rate, s, reward_total = [[0., 0.] for _ in range(3)], 0., 0, 0.

    log(rows, 0, steps, emit, **control_diagnostics(q, rate, reward_total, 0))
    for t in range(1, steps+1):
        # Tie break chooses action 0. Exploration ensures both actions are tried.
        a = rng.randrange(2) if rng.random() < .25 else max(range(2), key=lambda a: q[s][a])
        reward, sp = transition(s, a, rng)
        q, rate, delta = update(q, rate, s, a, reward, sp)

        reward_total += reward
        s = sp
        log(rows, t, steps, emit, **control_diagnostics(q, rate, reward_total, t))
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)

