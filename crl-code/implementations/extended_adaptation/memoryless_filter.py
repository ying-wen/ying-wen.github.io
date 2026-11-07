"""两状态HMM：隐状态以.9概率保持，观测以.8概率正确。
已知转移与观测模型；只估计belief，不学习这些参数，也没有RL控制更新。
测量为当前观测后真实隐状态的负log概率EMA；真实隐状态只用于测量。
忽略过去，均匀prior乘当前likelihood的无记忆对照。
"""
import math
import random
import torch
from torch import nn
try:
    from ._common import *
except ImportError:
    from _common import *
META = {
    "id": "extended-memoryless_filter",
    "name": "无记忆状态推断对照",
    "family": "extended_adaptation",
    "task": "two-state-hmm",
    "baseline": "extended-bayes_filter",
    "metric": "prequential_logloss_ema",
    "unit": "nats",
    "higher_better": False,
    "scope": "state-estimation",
    "implementation_kind": "component_experiment",
    "coverage_refs": [],
    "chapter_paths": [
        "construction/state/"
    ],
    "description": "无记忆状态推断对照的实际CPU教学实验；设置、公式和边界详见本文件。",
    "question": "实际学习或状态更新怎样改变此教学任务的独立测量？",
    "sources": [
        "https://people.eecs.berkeley.edu/~pabbeel/cs287-fa12/slides/bayes-filters.pdf"
    ],
    "defaults": {
        "steps": 600
    },
    "budget": "observations",
    "budget_note": "每 observation 一次状态条件化；无梯度 optimizer。",
    "limitations": "无记忆状态估计基线；value 是在线隐藏状态负对数损失的 EMA，不是冻结策略回报。隐藏状态仅用于诊断。"
}

def filter_step(belief,observation):
    # 步骤1：预测; 步骤2：观测校正; 步骤3：归一化。
    prior=[.5,.5]
    weighted=[prior[j]*(.8 if j==observation else .2) for j in range(2)]
    mass=sum(weighted)
    return [v/mass for v in weighted]

def run(seed=0,steps=600,emit=None):
    rng=setup(seed,steps); state=rng.randrange(2); belief=[.5,.5]
    rows=[]; ticks=grid(steps); score=math.log(2)
    log(rows,0,score,emit)
    for t in range(1,steps+1):
        state,observation=hmm_stream(rng,state)
        belief=filter_step(belief,observation)
        loss=-math.log(max(belief[state],1e-12))
        score=.95*score+.05*loss
        if t in ticks:log(rows,t,score,emit,posterior_one=belief[1],observation=observation)
    return rows

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META,run)
