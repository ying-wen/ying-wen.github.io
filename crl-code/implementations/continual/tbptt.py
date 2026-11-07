"""截断 BPTT。
每步8个U(-1,1)输入组成独立序列；tanh老师a=0.8，b中点从0.4变-0.4。学生初始[0.3,0.1,0]，段内参数冻结、段间SGD；曲线为更新前误差平方EMA(0.95)，每段隐藏状态清零。
"""
import random
import math
from pathlib import Path
try:
    from ._common import *
except ImportError:
    from _common import *

META = {
    "id": "tbptt",
    "name": "截断 BPTT",
    "family": "continual",
    "chapter_paths": [
        "construction/state/",
        "algorithms/credit-assignment/"
    ],
    "description": "每步8个U(-1,1)输入组成独立序列；tanh老师a=0.8，b中点从0.4变-0.4。学生初始[0.3,0.1,0]，段内参数冻结、段间SGD；曲线为更新前误差平方EMA(0.95)，每段隐藏状态清零。",
    "question": "精确Jacobian与三步截断怎样影响递归预测适应？",
    "task": "recurrent_sequence_prediction",
    "baseline": "rtrl",
    "metric": "更新前序列预测平方误差 EMA",
    "unit": "squared_error",
    "higher_better": False,
    "scope": "teaching-prediction",
    "sources": [
        "http://incompleteideas.net/book/the-book-2nd.html",
        "https://doi.org/10.1162/neco.1989.1.2.270"
    ],
    "defaults": {
        "steps": 1000
    },
    "budget": "sequences",
    "budget_note": "每步一段长度8序列；中点老师输入权重反转；段内参数冻结、段间更新；元素预算=8*steps。",
    "limitations": "小型教学任务，非论文基准复现；不声称持续学习算法的一般有效性。"
}

def gradient(theta,inputs,target,window=3):
    a,b,c=theta
    states=[0.]
    for x in inputs:
        states.append(math.tanh(a*states[-1]+b*x+c))
    error=states[-1]-target
    adjoint,grad=error,[0.,0.,0.]
    # 边界状态保留数值但detach梯度；不是把隐藏状态清零。
    left=max(0,len(inputs)-window)
    for t in reversed(range(left,len(inputs))):
        dz=adjoint*(1-states[t+1]**2)
        for j,feature in enumerate([states[t],inputs[t],1.]):
            grad[j]+=dz*feature
        adjoint=a*dz
    return grad,error

def update(theta,inputs,target,alpha=.05,window=3):
    grad,error=gradient(theta,inputs,target,window)
    return [w-alpha*g for w,g in zip(theta,grad)],error

def run(seed=0,steps=1000,emit=None):
    check_steps(steps)
    rng,rows,theta=random.Random(seed),[],[.3,.1,0.]
    error_ema=0.
    for t in range(1,steps+1):
        xs,target=recurrent_sequence(rng,t,steps)
        theta,error=update(theta,xs,target)
        error_ema=error*error if t==1 else .95*error_ema+.05*error*error
        log(rows,t,error_ema,steps,emit,recurrent_weight=theta[0],element_steps=8*t)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
