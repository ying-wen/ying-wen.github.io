"""Meta-gradient RL的lambda-return调参组件，固定策略随机游走价值学习。
内层：用lambda-return对5维表格V做一次SGD；外层：另一条新轨迹用固定
lambda'=1的目标衡量内层更新后的V，完整反传 through SGD 到sigmoid(beta)。
旧V的bootstrap数值detach，保留lambda对return的导数；不能对环境样本求导。
每批两条独立新轨迹，最多24转移/条，截断bootstrap不置terminal；真端点零尾。
固定lambda=.8对照，内/外样本和批次规模完全相同，不更新beta。
预算为训练批次，额外environment_steps诊断如实计数。
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
    "id": "extended-fixed_lambda",
    "name": "固定lambda-return对照",
    "family": "extended_adaptation",
    "task": "meta-random-walk",
    "baseline": "extended-meta_gradient",
    "metric": "frozen_value_mse",
    "unit": "squared_error",
    "higher_better": False,
    "scope": "teaching-prediction",
    "implementation_kind": "component_experiment",
    "coverage_refs": [],
    "chapter_paths": [
        "algorithms/meta/"
    ],
    "description": "固定lambda-return对照的实际CPU教学实验；设置、公式和边界详见本文件。",
    "question": "实际学习或状态更新怎样改变此教学任务的独立测量？",
    "sources": [
        "https://arxiv.org/abs/1805.09801"
    ],
    "defaults": {
        "steps": 600
    },
    "budget": "training_batches",
    "budget_note": "每批独立 train 和 validation 轨迹；一次价值内层更新，并计算元梯度；meta-gradient另更新元参数。转移数另记。",
    "limitations": "小型教学任务，不是原论文性能复现；组件实验不计作完整原方法。冻结评价与训练更新分离，预算不代表相同计算成本。"
}

def collect(rng):
    s=3; data=[]
    for _ in range(24):
        sp=s+(1 if rng.random()<.5 else -1)
        terminal=sp in (0,6); r=float(sp==6)
        data.append((s,r,sp,terminal))
        if terminal:break
        s=sp
    return data

def returns(theta,data,lam):
    old=theta.detach()
    last=data[-1]
    tail=torch.tensor(0.) if last[3] else old[last[2]-1]
    output=[]
    for s,r,sp,d in reversed(data):
        next_value=torch.tensor(0.) if d else old[sp-1]
        tail=r+(0 if d else 1)*((1-lam)*next_value+lam*tail)
        output.append(tail)
    return torch.stack(list(reversed(output)))

def meta_step(theta,beta,train,validation,meta_rate=.02):
    lam=beta.sigmoid()
    target=returns(theta,train,lam)
    index=torch.tensor([s-1 for s,r,sp,d in train])
    inner=.5*(theta[index]-target).square().mean()
    gradient=torch.autograd.grad(inner,theta,create_graph=True)[0]
    updated=theta-.1*gradient
    query_index=torch.tensor([s-1 for s,r,sp,d in validation])
    query_target=returns(updated,validation,torch.tensor(1.))
    outer=.5*(updated[query_index]-query_target.detach()).square().mean()
    meta_gradient=torch.autograd.grad(outer,beta)[0]
    return updated.detach().requires_grad_(),(beta-meta_rate*meta_gradient).detach().clamp(-4,4).requires_grad_(),float(outer.detach()),float(meta_gradient)

def run(seed=0,steps=600,emit=None):
    rng=setup(seed,steps); theta=torch.zeros(5,requires_grad=True)
    beta=torch.tensor(math.log(4.),requires_grad=True)
    truth=torch.arange(1,6)/6.; rows=[]; ticks=grid(steps); environment_steps=0
    log(rows,0,float((theta.detach()-truth).square().mean()),emit)
    for t in range(1,steps+1):
        train,validation=collect(rng),collect(rng); environment_steps+=len(train)+len(validation)
        theta,beta,outer,derivative=meta_step(theta,beta,train,validation,meta_rate=0.)
        if t in ticks:
            log(rows,t,float((theta.detach()-truth).square().mean()),emit,
                learned_lambda=float(beta.detach().sigmoid()),meta_gradient=derivative,heldout_loss=outer,
                environment_steps=environment_steps,training_batches=t)
    return rows

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META,run)
