"""Differential TD
两状态无终点交替流，从0开始，Bernoulli均值0.2/0.8，目标长期奖励率0.5、差分偏差差0.3。价值及率从0开始；原始转移预算。选项持续1或3步，学旧平均时长并归一化更新；只含固定策略预测子机制，非完整平均奖励选项控制。
δ=r−g+v(s')−v(s)，v(s)+=αδ，g+=ηαδ。
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
    "id": "differential_td_prediction",
    "name": "Differential TD",
    "family": "extended_classic",
    "coverage_refs": [
        "core-differential-td"
    ],
    "implementation_kind": "teaching_method",
    "task": "extended_average_prediction",
    "baseline": "average_smdp_option",
    "metric": "目标奖励率绝对误差",
    "unit": "reward_rate",
    "higher_better": False,
    "budget": "environment_steps",
    "scope": "teaching-average-reward-prediction",
    "chapter_paths": [
        "algorithms/average-reward/",
        "construction/options/"
    ],
    "sources": [
        "https://arxiv.org/abs/2006.16318"
    ],
    "defaults": {
        "steps": 1200
    },
    "description": "两状态无终点交替流，从0开始，Bernoulli均值0.2/0.8，目标长期奖励率0.5、差分偏差差0.3。价值及率从0开始；原始转移预算。选项持续1或3步，学旧平均时长并归一化更新；只含固定策略预测子机制，非完整平均奖励选项控制。",
    "question": "同一旧TD误差怎样同时估计差分价值和奖励率？",
    "limitations": "有限教学任务，不构成一般性能或原论文实验复现。"
}

def update(v,rate,s,reward,sp,alpha=.05,eta=.1):
    delta=reward-rate+v[sp]-v[s]
    new=v[:]
    new[s]+=alpha*delta
    return new,rate+eta*alpha*delta,delta

def run(seed=0,steps=1200,emit=None):
    check(steps)
    rng,rows,v,rate,s=random.Random(seed),[],[0.,0.],0.,0
    for t in range(1,steps+1):
        reward=float(rng.random()<[.2,.8][s])
        sp=1-s
        v,rate,delta=update(v,rate,s,reward,sp)
        s=sp
        log(rows,t,abs(rate-.5),steps,emit,bias_difference_error=abs(v[1]-v[0]-.3),rate=rate)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
