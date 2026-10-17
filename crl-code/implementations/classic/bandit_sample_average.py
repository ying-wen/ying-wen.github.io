"""样本均值 ε-greedy：独立、可运行的教学实现。
三臂平稳 Bernoulli 赌博机，均值为 0.2、0.5、0.8；记录实际采样奖励。
"""
import random
from pathlib import Path
try:
    from ._common import *
except ImportError:
    from _common import *

META = {
    "id": "bandit_sample_average",
    "name": "样本均值 ε-greedy",
    "family": "classic",
    "chapter_paths": [
        "algorithms/exploration/",
        "algorithms/control/",
        "foundations/objectives/"
    ],
    "description": "三臂平稳 Bernoulli 赌博机，均值为 0.2、0.5、0.8；记录实际采样奖励。",
    "question": "在同一教学任务和预算下，实际更新怎样改变预测或策略？",
    "task": "stationary_bernoulli_bandit",
    "baseline": "bandit_constant_step",
    "metric": "累计平均奖励",
    "unit": "reward",
    "higher_better": True,
    "scope": "teaching-control",
    "sources": [
        "http://incompleteideas.net/book/the-book-2nd.html"
    ],
    "defaults": {
        "steps": 1000
    },
    "budget": "environment_steps",
    "budget_note": "真实转移次数；未完成回合不当作终止；模型规划计算额外单列。",
    "initialization": "所有估值或偏好=0；三臂均值0.2/0.5/0.8；在线选择并更新。",
    "limitations": "小型教学任务；曲线不构成一般性能或持续学习优势的证据。"
}

def update(value, reward, count, alpha=None):
    # 样本均值用 1/N；固定步长保留对新奖励的持续敏感性。
    rate = 1/count if alpha is None else alpha
    return value+rate*(reward-value)

def run(seed=0, steps=1000, emit=None):
    check_steps(steps)
    rng = random.Random(seed)
    q, counts, rows = [0.]*3, [0]*3, []
    total, baseline = 0., 0.
    for t in range(1,steps+1):
        a = action(q,rng)
        reward = float(rng.random() < [.2,.5,.8][a])
        counts[a] += 1
        q[a] = update(q[a],reward,counts[a])
        total += reward
        baseline += (reward-baseline)/t
        log(rows,t,total/t,steps,emit)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
