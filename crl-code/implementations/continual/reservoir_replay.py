"""蓄水池经验回放。
流式两维线性预测：x=[1,U(-1,1)]，中点真权重从[0.2,0.7]变为[-0.2,-0.7]，观测噪声σ=0.03，初始w=0。每次更新后冻结测量当前真任务解析MSE。同时记录第一任务误差，保留与适应可能冲突。
"""
import random
import math
from pathlib import Path
try:
    from ._common import *
except ImportError:
    from _common import *

META = {
    "id": "reservoir_replay",
    "name": "蓄水池经验回放",
    "family": "continual",
    "chapter_paths": [
        "algorithms/retention/",
        "algorithms/plasticity/"
    ],
    "description": "流式两维线性预测：x=[1,U(-1,1)]，中点真权重从[0.2,0.7]变为[-0.2,-0.7]，观测噪声σ=0.03，初始w=0。每次更新后冻结测量当前真任务解析MSE。同时记录第一任务误差，保留与适应可能冲突。",
    "question": "旧任务保留与新任务适应怎样相互影响？",
    "task": "retention_regression",
    "baseline": "online_sgd",
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
    "budget_note": "每步一个真实样本；中点权重反转。EWC获知边界，回放额外一次梯度；非等计算、非等信息对照。",
    "limitations": "小型教学任务，非论文基准复现；不声称持续学习算法的一般有效性。"
}

def insert(buffer,item,seen,rng,capacity=32):
    # seen包含本条样本；每条历史样本有capacity/seen的保留概率。
    if len(buffer)<capacity:
        buffer.append(item)
    else:
        j=rng.randrange(seen)
        if j<capacity:
            buffer[j]=item

def update(w,x,target,alpha=.03):
    error=target-dot(w,x)
    return [wi+alpha*error*xi for wi,xi in zip(w,x)],error

def run(seed=0,steps=1000,emit=None):
    check_steps(steps)
    rng,rows,w=random.Random(seed),[],[0.,0.]
    buffer=[]
    replay_rng=random.Random(seed+100000)
    for t in range(1,steps+1):
        x,y,target=signal(rng,t,steps)
        w,error=update(w,x,y)
        insert(buffer,(x[:],y),t,replay_rng)
        # 每个流样本额外一次回放；多一次计算不是免费的优势。
        rx,ry=replay_rng.choice(buffer)
        w,_=update(w,rx,ry)
        log(rows,t,regression_error(w,target),steps,emit,
            old_task_mse=regression_error(w,[.2,.7]),prediction_error=error)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
