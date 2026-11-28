"""True-online TD(λ)：独立、可运行的教学实现。
五个非终止状态的无偏随机游走；左端奖励 0、右端奖励 1，γ=1。
"""
import random
from pathlib import Path
try:
    from ._common import *
except ImportError:
    from _common import *

META = {
    "id": "true_online_td",
    "name": "True-online TD(λ)",
    "family": "classic",
    "chapter_paths": [
        "algorithms/credit-assignment/",
        "algorithms/value/",
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

def update(v,traces,s,reward,sp,old_prediction,alpha=.1,lam=.8):
    current = v[s]
    next_prediction = v[sp] if sp is not None else 0.
    delta = reward+next_prediction-current
    # Dutch trace 的内积在衰减前读取；one-hot 特征的内积就是 e[s]。
    correction = 1-alpha*lam*traces[s]
    for j in traces:
        traces[j] *= lam
    traces[s] += correction
    # True-online 的两项修正不可省略。
    for j in v:
        v[j] += alpha*(delta+current-old_prediction)*traces[j]
    v[s] -= alpha*(current-old_prediction)
    return next_prediction

def run(seed=0,steps=1000,emit=None):
    check_steps(steps)
    rng, rows = random.Random(seed), []
    v, s = dict.fromkeys(range(1,6),0.), 3
    traces, old_prediction = dict.fromkeys(range(1,6),0.), 0.
    for t in range(1,steps+1):
        reward, sp = walk(s,rng)
        old_prediction = update(v,traces,s,reward,sp,old_prediction)
        if sp is None:
            traces, old_prediction = dict.fromkeys(range(1,6),0.), 0.
        s = 3 if sp is None else sp
        log(rows,t,rmse(v),steps,emit)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
