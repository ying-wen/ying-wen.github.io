"""Differential Dyna · learned empirical model
每个真实转移更新Q与g并学习奖励/转移频数，再做5次经验模型期望备份。模型只含已访问状态动作。环境步和额外规划计算分别记录。
"""
import random
from pathlib import Path
try:
    from ._common import *
except ImportError:
    from _common import *

META = {
    "id": "differential_dyna",
    "name": "Differential Dyna · learned empirical model",
    "family": "average_systems",
    "implementation_kind": "component_experiment",
    "coverage_refs": [],
    "task": "average_control_learned_model",
    "baseline": "differential_q_multistate",
    "metric": "冻结贪心策略的精确平均奖励",
    "unit": "reward_rate",
    "higher_better": True,
    "budget": "environment_steps",
    "scope": "empirical-model-dyna-experiment",
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
    "description": "每个真实转移更新Q与g并学习奖励/转移频数，再做5次经验模型期望备份。模型只含已访问状态动作。环境步和额外规划计算分别记录。",
    "question": "学到的模型能否减少真实样本需求，误差与规划计算又如何计量？",
    "limitations": "经验模型、常数步长和规划调度的有限任务实验；不是原论文完整复现，不宣称非平稳或非线性收敛。相同环境步不代表相同计算量。"
}


def update(q, rate, s, a, reward, sp, alpha=.08, eta=.1):
    # Off-policy control: conditional (s,a) dynamics already correct; no pi/b.
    delta = reward-rate+max(q[sp])-q[s][a]
    new = [row[:] for row in q]
    new[s][a] += alpha*delta
    return new, rate+eta*alpha*delta, delta


def model_update(model, s, a, reward, sp):
    entry = model.setdefault((s, a), {'count': 0, 'reward_sum': 0., 'next': [0, 0, 0]})
    entry['count'] += 1
    entry['reward_sum'] += reward
    entry['next'][sp] += 1


def planning_update(q, rate, pair, entry, alpha=.08, eta=.1):
    # Expected ONE state-action backup; it sums over three possible next states.
    # The empirical model is learned from real observations, not true P.
    n = entry['count']
    if n <= 0:
        raise ValueError('Never plan from an unobserved state-action pair')
    s, a = pair
    delta = entry['reward_sum']/n-rate + sum(entry['next'][sp]/n*max(q[sp])
                                             for sp in range(3))-q[s][a]
    new = [row[:] for row in q]
    new[s][a] += alpha*delta
    return new, rate+eta*alpha*delta, delta


def model_error(model):
    if not model:
        return 0.
    return sum(sum(abs(entry['next'][sp]/entry['count']-P[s][a][sp]) for sp in range(3))
               for (s, a), entry in model.items())/len(model)


def run(seed=0, steps=1200, emit=None):
    check_steps(steps)
    rng, rows = random.Random(seed), []
    q, rate, s, reward_total = [[0., 0.] for _ in range(3)], 0., 0, 0.
    planner_rng, model, backups = random.Random(seed+100003), {}, 0
    log(rows, 0, steps, emit, **control_diagnostics(q, rate, reward_total, 0), observed_pairs=0, transition_l1_error=0.)
    for t in range(1, steps+1):
        # Tie break chooses action 0. Exploration ensures both actions are tried.
        a = rng.randrange(2) if rng.random() < .25 else max(range(2), key=lambda a: q[s][a])
        reward, sp = transition(s, a, rng)
        q, rate, delta = update(q, rate, s, a, reward, sp)
        model_update(model, s, a, reward, sp)
        for _ in range(5):
            pair = planner_rng.choice(sorted(model))
            q, rate, delta = planning_update(q, rate, pair, model[pair])
            backups += 1
        reward_total += reward
        s = sp
        log(rows, t, steps, emit, **control_diagnostics(q, rate, reward_total, t, backups), observed_pairs=len(model), transition_l1_error=model_error(model))
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)

