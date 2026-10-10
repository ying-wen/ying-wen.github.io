"""TIDBD(λ)。
流式两维线性预测：x=[1,U(-1,1)]，中点真权重从[0.2,0.7]变为[-0.2,-0.7]，观测噪声σ=0.03，初始w=0。每次更新后冻结测量当前真任务解析MSE。此γ=0退化教学仅验证即时预测步长自适应，不检验带bootstrap的TIDBD。
"""
import random
import math
from pathlib import Path
try:
    from ._common import *
except ImportError:
    from _common import *

META = {
    "id": "tidbd",
    "name": "TIDBD(λ)",
    "family": "continual",
    "chapter_paths": [
        "algorithms/meta/",
        "algorithms/value/",
        "algorithms/streaming/"
    ],
    "description": "流式两维线性预测：x=[1,U(-1,1)]，中点真权重从[0.2,0.7]变为[-0.2,-0.7]，观测噪声σ=0.03，初始w=0。每次更新后冻结测量当前真任务解析MSE。此γ=0退化教学仅验证即时预测步长自适应，不检验带bootstrap的TIDBD。",
    "question": "逐特征步长怎样响应线性预测权重反转？",
    "task": "adaptive_regression",
    "baseline": "constant_step_lms",
    "metric": "当前任务无噪声预测 MSE",
    "unit": "squared_error",
    "higher_better": False,
    "scope": "teaching-prediction",
    "sources": [
        "http://incompleteideas.net/book/the-book-2nd.html",
        "https://arxiv.org/abs/1804.03334"
    ],
    "defaults": {
        "steps": 1000
    },
    "budget": "environment_steps",
    "budget_note": "每步一个真实样本；中点权重反转，无任务标签；TIDBD采用γ=0退化协议。",
    "limitations": "小型教学任务，非论文基准复现；不声称持续学习算法的一般有效性。"
}

def update(w,beta,history,trace,x,reward,xp,gamma=0.,lam=.8,meta_rate=.01):
    # 2018半梯度TIDBD版本；本实验γ=0，退化为监督即时预测。
    # 保留一般γ和λ公式以便手算，但不声称此实验验证完整TD自适应。
    delta = reward+gamma*dot(w,xp)-dot(w,x)
    new_beta = [b+meta_rate*delta*xi*hi for b,xi,hi in zip(beta,x,history)]
    rates = [math.exp(b) for b in new_beta]
    z = [gamma*lam*zi+xi for zi,xi in zip(trace,x)]
    new_w = [wi+a*delta*zi for wi,a,zi in zip(w,rates,z)]
    new_h = [hi*max(0.,1-a*xi*zi)+a*delta*zi for hi,a,xi,zi in zip(history,rates,x,z)]
    return new_w,new_beta,new_h,z,delta

def run(seed=0,steps=1000,emit=None):
    check_steps(steps)
    rng,rows,w=random.Random(seed),[],[0.,0.]
    beta,history,trace = [math.log(.03)]*2,[0.,0.],[0.,0.]
    for t in range(1,steps+1):
        x,y,target=signal(rng,t,steps)
        w,beta,history,trace,error = update(w,beta,history,trace,x,y,[0.,0.])
        log(rows,t,regression_error(w,target),steps,emit,
            old_task_mse=regression_error(w,[.2,.7]),prediction_error=error)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
