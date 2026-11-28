"""IDBD。
流式两维线性预测：x=[1,U(-1,1)]，中点真权重从[0.2,0.7]变为[-0.2,-0.7]，观测噪声σ=0.03，初始w=0。每次更新后冻结测量当前真任务解析MSE。
"""
import random
import math
from pathlib import Path
try:
    from ._common import *
except ImportError:
    from _common import *

META = {
    "id": "idbd",
    "name": "IDBD",
    "family": "continual",
    "chapter_paths": [
        "algorithms/meta/",
        "algorithms/value/",
        "algorithms/streaming/"
    ],
    "description": "流式两维线性预测：x=[1,U(-1,1)]，中点真权重从[0.2,0.7]变为[-0.2,-0.7]，观测噪声σ=0.03，初始w=0。每次更新后冻结测量当前真任务解析MSE。",
    "question": "逐特征步长怎样响应线性预测权重反转？",
    "task": "adaptive_regression",
    "baseline": "constant_step_lms",
    "metric": "当前任务无噪声预测 MSE",
    "unit": "squared_error",
    "higher_better": False,
    "scope": "teaching-prediction",
    "sources": [
        "http://incompleteideas.net/book/the-book-2nd.html",
        "https://cdn.aaai.org/AAAI/1992/AAAI92-027.pdf"
    ],
    "defaults": {
        "steps": 1000
    },
    "budget": "environment_steps",
    "budget_note": "每步一个真实样本；中点权重反转，无任务标签；TIDBD采用γ=0退化协议。",
    "limitations": "小型教学任务，非论文基准复现；不声称持续学习算法的一般有效性。"
}

def update(w,beta,history,x,target,meta_rate=.01):
    error = target-dot(w,x)
    # 第一步：误差与旧对角敏感度决定每个特征的log步长。
    new_beta = [b+meta_rate*error*xi*hi for b,xi,hi in zip(beta,x,history)]
    rates = [math.exp(b) for b in new_beta]
    # 第二步：新步长更新权重；第三步：更新本地敏感度并裁剪负衰减。
    new_w = [wi+a*error*xi for wi,a,xi in zip(w,rates,x)]
    new_h = [hi*max(0.,1-a*xi*xi)+a*error*xi for hi,a,xi in zip(history,rates,x)]
    return new_w,new_beta,new_h,error

def run(seed=0,steps=1000,emit=None):
    check_steps(steps)
    rng,rows,w=random.Random(seed),[],[0.,0.]
    beta,history,trace = [math.log(.03)]*2,[0.,0.],[0.,0.]
    for t in range(1,steps+1):
        x,y,target=signal(rng,t,steps)
        w,beta,history,error = update(w,beta,history,x,y)
        log(rows,t,regression_error(w,target),steps,emit,
            old_task_mse=regression_error(w,[.2,.7]),prediction_error=error)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
