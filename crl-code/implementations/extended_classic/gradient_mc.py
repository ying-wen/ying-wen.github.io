"""Gradient MC
无偏五状态随机游走，从3开始，γ=1，特征[1,s/6]、w=0。冻结当前线性预测测量五状态RMSE；LSTD显式1e-5对角正则，MC只更新完整回合。
w←w+α(G−w·x)x；完整回合回报作为固定标签。
时序、边界与未覆盖范围见核心注释及 docs/extended-classic.md。
"""
import math
import random
from pathlib import Path
try:
    from ._common import *
except ImportError:
    from _common import *

META = {
    "id": "gradient_mc",
    "name": "Gradient MC",
    "family": "extended_classic",
    "coverage_refs": [
        "core-gradient-mc"
    ],
    "implementation_kind": "teaching_method",
    "task": "extended_linear_walk",
    "baseline": "lstd",
    "metric": "五状态线性价值 RMSE",
    "unit": "value",
    "higher_better": False,
    "budget": "environment_steps",
    "scope": "teaching-prediction",
    "chapter_paths": [
        "foundations/approximation/prediction/",
        "algorithms/value/"
    ],
    "sources": [
        "http://incompleteideas.net/book/the-book-2nd.html"
    ],
    "defaults": {
        "steps": 1200
    },
    "description": "无偏五状态随机游走，从3开始，γ=1，特征[1,s/6]、w=0。冻结当前线性预测测量五状态RMSE；LSTD显式1e-5对角正则，MC只更新完整回合。",
    "question": "真实回报平方损失的梯度怎样学习共享线性预测？",
    "limitations": "有限教学任务，不构成一般性能或原论文实验复现。"
}

def update(w,x,target,alpha=.03):
    # 标签是完整真实回报，对当前预测求真正梯度。
    error=target-dot(w,x)
    return [wi+alpha*error*xi for wi,xi in zip(w,x)]

def run(seed=0,steps=1200,emit=None):
    check(steps)
    rng,rows,w,s=random.Random(seed),[],[0.,0.],3
    states,rewards=[],[]
    for t in range(1,steps+1):
        r,sp=walk(s,rng)
        states.append(s)
        rewards.append(r)
        if sp is None:
            g=0.
            for j in reversed(range(len(rewards))):
                g=rewards[j]+g
                w=update(w,features(states[j]),g)
            states,rewards=[],[]
        s=3 if sp is None else sp
        log(rows,t,linear_error(w),steps,emit)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
