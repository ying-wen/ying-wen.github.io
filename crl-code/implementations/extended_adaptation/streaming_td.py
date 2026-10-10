"""两状态交替继续预测，gamma=.8，奖励到达1为amplitude，幅度中点1->.5。
非线性网络与累积梯度迹在每条转移实际更新，不缓存样本，不用target网络。
固定alpha=.03 TD(lambda)对照，与ObGD同网络/迹/目标/数据流。
目标使用更新前网络bootstrap并停止梯度；trace q=.8*.8，无终止或伪reset。
此步长缩放不是任意非线性网络的全局不发散定理。
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
    "id": "extended-streaming_td",
    "name": "固定步长非线性TD对照",
    "family": "extended_adaptation",
    "task": "nonlinear-alternating-prediction",
    "baseline": "extended-obgd2024",
    "metric": "frozen_value_mse",
    "unit": "squared_error",
    "higher_better": False,
    "scope": "teaching-prediction",
    "implementation_kind": "teaching_method",
    "coverage_refs": [],
    "chapter_paths": [
        "algorithms/streaming/"
    ],
    "description": "固定步长非线性TD对照的实际CPU教学实验；设置、公式和边界详见本文件。",
    "question": "实际学习或状态更新怎样改变此教学任务的独立测量？",
    "sources": [
        "https://github.com/mohmdelsayed/streaming-drl/blob/main/optim.py"
    ],
    "defaults": {
        "steps": 600
    },
    "budget": "environment_steps",
    "budget_note": "按实际环境转移计步；各算法更新/回放频率见本文件循环，环境步相同不等于相同计算成本。",
    "limitations": "小型教学任务，不是原论文性能复现；组件实验不计作完整原方法。冻结评价与训练更新分离，预算不代表相同计算成本。"
}

def step(parameters,trace,gradient,delta,alpha=.03,q=.64,kappa=2.):
    trace=q*trace+gradient
    effective=alpha
    with torch.no_grad():
        offset=0
        for p in parameters:
            n=p.numel(); p.add_((effective*delta*trace[offset:offset+n]).reshape_as(p)); offset+=n
    return trace,effective

def run(seed=0,steps=600,emit=None):
    setup(seed,steps); net=mlp(2,1); parameters=list(net.parameters())
    trace=torch.zeros(sum(p.numel() for p in parameters))
    rows=[]; ticks=grid(steps); state=0
    def error(amplitude):
        with torch.no_grad():return alternating_value_error(net(torch.eye(2)).squeeze(-1),amplitude)
    log(rows,0,error(1.),emit)
    for t in range(1,steps+1):
        amplitude=1. if t<=steps//2 else .5
        next_state=1-state
        x=torch.eye(2)[state]; xp=torch.eye(2)[next_state]
        prediction=net(x).squeeze(-1)
        with torch.no_grad():delta=float(amplitude*float(next_state==1)+.8*net(xp).squeeze(-1)-prediction)
        gradient=torch.cat([g.reshape(-1) for g in torch.autograd.grad(prediction,parameters)])
        trace,effective=step(parameters,trace,gradient,delta)
        state=next_state
        if t in ticks:
            log(rows,t,error(amplitude),emit,phase="before_change" if t<=steps//2 else "after_change",
                step_size=effective,td_error=delta,trace_l1=float(trace.abs().sum()))
    return rows

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META,run)
