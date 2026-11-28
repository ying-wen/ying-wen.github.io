"""对角 EWC。
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
    "id": "ewc",
    "name": "对角 EWC",
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

def update(w,x,target,anchor,fisher,strength=.5,alpha=.03):
    error = target-dot(w,x)
    # 当前平方损失梯度加冻结旧任务锚点的二次罚项梯度。
    return [wi+alpha*(error*xi-strength*fi*(wi-ai))
            for wi,xi,ai,fi in zip(w,x,anchor,fisher)],error

def run(seed=0,steps=1000,emit=None):
    check_steps(steps)
    rng,rows,w=random.Random(seed),[],[0.,0.]
    anchor,fisher,information = [0.,0.],[0.,0.],[0.,0.]
    for t in range(1,steps+1):
        x,y,target=signal(rng,t,steps)
        if t == steps//2+1:
            anchor = w[:]
            # 方差1的线性Gaussian工作模型的对角Fisher为E[x_i²]。
            fisher = [v/(steps//2) for v in information]
        w,error = update(w,x,y,anchor,fisher)
        if t<=steps//2:
            information = [v+xi*xi for v,xi in zip(information,x)]
        log(rows,t,regression_error(w,target),steps,emit,
            old_task_mse=regression_error(w,[.2,.7]),prediction_error=error)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
