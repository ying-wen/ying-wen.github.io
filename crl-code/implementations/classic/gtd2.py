"""GTD2：独立、可运行的教学实现。
在线随机游走预测，从状态3开始；γ=1，特征[1,s/6]，冻结测量五状态RMSE。全部使用同策略数据ρ=1，因此此实验不检验离策略稳定性。
"""
import random
from pathlib import Path
try:
    from ._common import *
except ImportError:
    from _common import *

META = {
    "id": "gtd2",
    "name": "GTD2",
    "family": "classic",
    "chapter_paths": [
        "algorithms/value/",
        "algorithms/streaming/",
        "foundations/objectives/"
    ],
    "description": "在线随机游走预测，从状态3开始；γ=1，特征[1,s/6]，冻结测量五状态RMSE。全部使用同策略数据ρ=1，因此此实验不检验离策略稳定性。",
    "question": "在同一教学任务和预算下，实际更新怎样改变预测或策略？",
    "task": "random_walk_linear_prediction",
    "baseline": "linear_td",
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

def update(w,h,x,reward,xp,alpha=.03,beta=.1):
    # h 与 w 同时读取旧参数。ρ=1，γ=1；终点 xp 是零向量。
    delta = reward+dot(w,xp)-dot(w,x)
    inner = dot(h,x)
    new_w = [wi+alpha*(xi-xpi)*inner for wi,xi,xpi in zip(w,x,xp)]
    new_h = [hi+beta*(delta-inner)*xi for hi,xi in zip(h,x)]
    return new_w,new_h,delta

def run(seed=0,steps=1000,emit=None):
    check_steps(steps)
    rng, rows, w, s = random.Random(seed), [], [0.,0.], 3
    h = [0.,0.]
    for t in range(1,steps+1):
        reward,sp = walk(s,rng)
        x,xp = features(s),features(sp)
        w,h,delta = update(w,h,x,reward,xp)
        s = 3 if sp is None else sp
        log(rows,t,linear_error(w),steps,emit,td_error=delta)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
