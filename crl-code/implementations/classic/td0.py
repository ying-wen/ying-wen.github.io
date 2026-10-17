"""TD(0)：独立、可运行的教学实现。
五个非终止状态的无偏随机游走；左端奖励 0、右端奖励 1，γ=1。
"""
import random
from pathlib import Path
try:
    from ._common import *
except ImportError:
    from _common import *

META = {
    "id": "td0",
    "name": "TD(0)",
    "family": "classic",
    "chapter_paths": [
        "algorithms/value/",
        "algorithms/credit-assignment/",
        "foundations/objectives/"
    ],
    "description": "五个非终止状态的无偏随机游走；左端奖励 0、右端奖励 1，γ=1。",
    "question": "在同一教学任务和预算下，实际更新怎样改变预测或策略？",
    "task": "random_walk_prediction",
    "baseline": "mc_prediction",
    "metric": "五状态价值 RMSE",
    "unit": "value",
    "higher_better": False,
    "scope": "teaching-prediction",
    "sources": [
        "http://incompleteideas.net/book/the-book-2nd.html"
    ],
    "defaults": {
        "steps": 1000
    },
    "budget": "environment_steps",
    "budget_note": "真实转移次数；未完成回合不当作终止；模型规划计算额外单列。",
    "initialization": "价值或权重=0；每回合从状态3开始，无偏左右随机游走，在线预测。",
    "limitations": "小型教学任务；曲线不构成一般性能或持续学习优势的证据。"
}

def update(v,s,reward,sp,alpha=.1):
    # 真终点价值为零；预算截断不冒充终点。
    delta = reward+(v[sp] if sp is not None else 0.)-v[s]
    v[s] += alpha*delta
    return delta

def run(seed=0,steps=1000,emit=None):
    check_steps(steps)
    rng, rows = random.Random(seed), []
    v, s = dict.fromkeys(range(1,6),0.), 3
    for t in range(1,steps+1):
        reward, sp = walk(s,rng)
        update(v,s,reward,sp)
        s = 3 if sp is None else sp
        log(rows,t,rmse(v),steps,emit)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
