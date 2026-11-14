"""无回放 actor-critic：CLEAR 的同任务新样本基线，终止 TD 目标为即时奖励。"""
import math
import random
import torch
from torch import nn
try:
    from ._common import *
except ImportError:
    from _common import *

META = {
    "id": "extended-fresh_ac",
    "name": "无回放 Actor-Critic",
    "family": "extended_adaptation",
    "task": "context-bandit",
    "baseline": "extended-clear",
    "metric": "冻结当前上下文策略期望奖励",
    "unit": "return",
    "higher_better": True,
    "scope": "teaching-control",
    "implementation_kind": "teaching_method",
    "coverage_refs": [],
    "chapter_paths": [
        "algorithms/meta/"
    ],
    "description": "无回放 Actor-Critic的实际CPU教学实验；设置、公式和边界详见本文件。",
    "question": "实际学习或状态更新怎样改变此教学任务的独立测量？",
    "sources": [
        "https://arxiv.org/abs/1811.11682"
    ],
    "defaults": {
        "steps": 600
    },
    "budget": "environment_steps",
    "budget_note": "按实际环境转移计步；各算法更新/回放频率见本文件循环，环境步相同不等于相同计算成本。",
    "limitations": "小型教学任务，不是原论文性能复现；组件实验不计作完整原方法。冻结评价与训练更新分离，预算不代表相同计算成本。"
}

def reservoir(buffer,item,t,rng,capacity=64):
    if len(buffer)<capacity: buffer.append(item)
    else:
        j=rng.randrange(t)
        if j<capacity: buffer[j]=item

def replay_loss(actor,critic,item,clone=True):
    x,a,r,old_p,old_v=item
    p=actor(x).softmax(0); v=critic(x).squeeze()
    # 真终止，目标没有 gamma V(s')；比率/优势 stop-gradient。
    rho=(p[a].detach()/old_p[a]).clamp(max=1.)
    delta=(r-v.detach())
    loss=.5*(v-(v.detach()+rho*delta)).square()-rho*delta*p[a].log()
    if clone:
        loss+=.1*(old_p*(old_p.log()-p.log())).sum()+.1*(v-old_v).square()
    return loss

def run(seed=0,steps=600,emit=None):
    rng=setup(seed,steps); replay_rng=random.Random(seed+551)
    actor=mlp(2,2); critic=mlp(2,1)
    opt=torch.optim.SGD(list(actor.parameters())+list(critic.parameters()),lr=.05)
    buffer=[]; rows=[]; points=grid(steps)
    def evaluate(t):
        with torch.no_grad():
            p0=actor(torch.tensor([1.,0.])).softmax(0)[1]
            p1=actor(torch.tensor([0.,1.])).softmax(0)[0]
        log(rows,t,p1 if t>steps//2 else p0,emit,old_return=float(p0),
            replay_size=len(buffer),optimizer_steps=t,environment_steps=t)
    evaluate(0)
    for t in range(1,steps+1):
        context=int(t>steps//2); x=torch.tensor([1.-context,float(context)])
        with torch.no_grad():
            p=actor(x).softmax(0); old_v=critic(x).squeeze().clone()
        a=int(rng.random()<float(p[1])); r=float(a==1-context)
        item=(x,a,r,p.clone(),old_v)
        # fresh 的 on-policy actor/critic；克隆仅旧样本，避免伪自监督。
        loss=replay_loss(actor,critic,item,False)
        if False: loss=.5*loss
        opt.zero_grad(); loss.backward(); opt.step()
        # 无回放基线不存储历史行为样本。
        if t in points:evaluate(t)
    return rows

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META,run)
