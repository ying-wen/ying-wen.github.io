"""CLEAR 的终止上下文 bandit 组件实验。
步骤1：先存行为分布和行为价值，再更新；步骤2：蓄水池保持有限历史样本；
步骤3：旧样本使用截断重要性比 TD 与策略梯度，并加入行为 KL 和价值克隆。
这里每次拉杆即终止，V-trace 退化为单步；没有 IMPALA 多步/分布式系统。
上下文0到1切换不改变旧任务，old_return 真正测旧上下文；主指标是当前上下文
随机策略精确期望奖励，不是训练奖励。相同环境步不表示相同梯度计算成本。
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
    "id": "extended-clear",
    "name": "CLEAR 单步回放组件",
    "family": "extended_adaptation",
    "task": "context-bandit",
    "baseline": "extended-fresh_ac",
    "metric": "冻结当前上下文策略期望奖励",
    "unit": "return",
    "higher_better": True,
    "scope": "teaching-control",
    "implementation_kind": "component_experiment",
    "coverage_refs": [
        "core-clear"
    ],
    "chapter_paths": [
        "algorithms/meta/"
    ],
    "description": "CLEAR 单步回放组件的实际CPU教学实验；设置、公式和边界详见本文件。",
    "question": "实际学习或状态更新怎样改变此教学任务的独立测量？",
    "sources": [
        "https://arxiv.org/abs/1811.11682"
    ],
    "defaults": {
        "steps": 600
    },
    "budget": "environment_steps",
    "budget_note": "按实际环境转移计步；各算法更新/回放频率见本文件循环，环境步相同不等于相同计算成本。",
    "limitations": "终止上下文bandit的单步V-trace/策略与价值克隆组件，有蓄水池；无多步分布式CLEAR系统。current return是随机策略期望，old_return为旧上下文诊断。"
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
        if buffer: loss=.5*loss+.5*replay_loss(actor,critic,replay_rng.choice(buffer))
        opt.zero_grad(); loss.backward(); opt.step()
        reservoir(buffer,item,t,replay_rng)
        if t in points:evaluate(t)
    return rows

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META,run)
