"""三步 TD 预测：独立、可运行的教学实现。
五个非终止状态的无偏随机游走；左端奖励 0、右端奖励 1，γ=1。
"""
import random
from pathlib import Path
try:
    from ._common import *
except ImportError:
    from _common import *

META = {
    "id": "nstep_td",
    "name": "三步 TD 预测",
    "family": "classic",
    "chapter_paths": [
        "algorithms/value/",
        "algorithms/credit-assignment/",
        "foundations/objectives/"
    ],
    "description": "五个非终止状态的无偏随机游走；左端奖励 0、右端奖励 1，γ=1。",
    "question": "在同一教学任务和预算下，实际更新怎样改变预测或策略？",
    "task": "random_walk_prediction",
    "baseline": "td0",
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

def update(v,state,rewards,bootstrap_state,alpha=.1):
    # γ=1 的 n-step 目标；bootstrap_state=None 仅表示真实终点。
    target = sum(rewards)+(v[bootstrap_state] if bootstrap_state is not None else 0.)
    v[state] += alpha*(target-v[state])
    return target

def run(seed=0,steps=1000,emit=None):
    check_steps(steps)
    rng, rows = random.Random(seed), []
    v, s = dict.fromkeys(range(1,6),0.), 3
    pending = []
    for t in range(1,steps+1):
        reward, sp = walk(s,rng)
        pending.append((s,reward))
        if len(pending) >= 3:
            update(v,pending[0][0],[r for _,r in pending[:3]],sp)
            pending.pop(0)
        if sp is None:
            while pending:
                update(v,pending[0][0],[r for _,r in pending],None)
                pending.pop(0)
        # 预算末尾尚未成熟的目标保留为未更新，不能伪造终点。
        s = 3 if sp is None else sp
        log(rows,t,rmse(v),steps,emit)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
