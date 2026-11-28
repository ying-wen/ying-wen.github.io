"""固定步长 LMS。
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
    "id": "constant_step_lms",
    "name": "固定步长 LMS",
    "family": "continual",
    "chapter_paths": [
        "algorithms/meta/",
        "algorithms/value/",
        "algorithms/streaming/"
    ],
    "description": "流式两维线性预测：x=[1,U(-1,1)]，中点真权重从[0.2,0.7]变为[-0.2,-0.7]，观测噪声σ=0.03，初始w=0。每次更新后冻结测量当前真任务解析MSE。",
    "question": "逐特征步长怎样响应线性预测权重反转？",
    "task": "adaptive_regression",
    "baseline": "idbd",
    "metric": "当前任务无噪声预测 MSE",
    "unit": "squared_error",
    "higher_better": False,
    "scope": "teaching-prediction",
    "sources": [
        "http://incompleteideas.net/book/the-book-2nd.html"
    ],
    "defaults": {
        "steps": 1000
    },
    "budget": "environment_steps",
    "budget_note": "每步一个真实样本；中点权重反转，无任务标签；TIDBD采用γ=0退化协议。",
    "limitations": "小型教学任务，非论文基准复现；不声称持续学习算法的一般有效性。"
}

def update(w,x,target,alpha=.03):
    error=target-dot(w,x)
    return [wi+alpha*error*xi for wi,xi in zip(w,x)],error

def run(seed=0,steps=1000,emit=None):
    check_steps(steps)
    rng,rows,w=random.Random(seed),[],[0.,0.]
    for t in range(1,steps+1):
        x,y,target=signal(rng,t,steps)
        w,error=update(w,x,y)
        log(rows,t,regression_error(w,target),steps,emit,
            old_task_mse=regression_error(w,[.2,.7]),prediction_error=error)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
