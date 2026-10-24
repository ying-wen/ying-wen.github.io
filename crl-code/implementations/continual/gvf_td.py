"""向量 GVF TD。
继续两状态确定性交替流，从0开始，无终点。两个cumulant分别为到达1和到达0；折扣0.8/0.5，中点cumulant幅度从1变0.5。one-hot权重全0，在线更新后冻结测量解析GVF RMSE；ρ=1，不检验离策略稳定性。
"""
import random
import math
from pathlib import Path
try:
    from ._common import *
except ImportError:
    from _common import *

META = {
    "id": "gvf_td",
    "name": "向量 GVF TD",
    "family": "continual",
    "chapter_paths": [
        "construction/predictive-knowledge/",
        "algorithms/value/"
    ],
    "description": "继续两状态确定性交替流，从0开始，无终点。两个cumulant分别为到达1和到达0；折扣0.8/0.5，中点cumulant幅度从1变0.5。one-hot权重全0，在线更新后冻结测量解析GVF RMSE；ρ=1，不检验离策略稳定性。",
    "question": "不同折扣GVF怎样响应cumulant减半？",
    "task": "vector_gvf_prediction",
    "baseline": "gvf_gtd_lambda",
    "metric": "两个 GVF 四状态分量 RMSE",
    "unit": "value",
    "higher_better": False,
    "scope": "teaching-prediction",
    "sources": [
        "http://incompleteideas.net/book/the-book-2nd.html",
        "https://sites.ualberta.ca/~pilarski/docs/papers/Sutton_2011_Horde_AAMAS.pdf"
    ],
    "defaults": {
        "steps": 1000
    },
    "budget": "environment_steps",
    "budget_note": "每步一个继续转移；中点cumulant幅度减半，ρ=1，仅on-policy教学。",
    "limitations": "小型教学任务，非论文基准复现；不声称持续学习算法的一般有效性。"
}

def update(weights,s,sp,cumulants,gammas,alpha=.05):
    new=[w[:] for w in weights]
    for k,(c,gamma) in enumerate(zip(cumulants,gammas)):
        delta=c+gamma*weights[k][sp]-weights[k][s]
        new[k][s]+=alpha*delta
    return new

def run(seed=0,steps=1000,emit=None):
    check_steps(steps)
    rows,weights,s=[],[[0.,0.],[0.,0.]],0
    gammas=[.8,.5]
    for t in range(1,steps+1):
        amplitude=1. if t<=steps//2 else .5
        sp=1-s
        cumulants=[amplitude*float(sp==1),amplitude*float(sp==0)]
        weights=update(weights,s,sp,cumulants,gammas)
        truths=[[amplitude/(1-.8**2),.8*amplitude/(1-.8**2)],
                [.5*amplitude/(1-.5**2),amplitude/(1-.5**2)]]
        error=math.sqrt(sum((weights[k][j]-truths[k][j])**2 for k in range(2) for j in range(2))/4)
        s=sp
        log(rows,t,error,steps,emit)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
