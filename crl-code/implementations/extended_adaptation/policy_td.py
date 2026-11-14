"""单工作器 actor-critic，固定旧价值与旧策略采样批次。
gamma=1；12步截止是真终止，批次截止仍使用 next value 且不重置环境。
loss=-mean(log pi * 固定优势)，策略梯度不穿过优势。
采样数据在更新后清空，预算末不完整Monte Carlo回合不得伪装零尾价值。
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
    "id": "extended-policy_td",
    "name": "一步Actor-Critic对照",
    "family": "extended_adaptation",
    "task": "extended-chain",
    "baseline": "extended-reinforce",
    "metric": "frozen_greedy_return",
    "unit": "return",
    "higher_better": True,
    "scope": "teaching-control",
    "implementation_kind": "teaching_method",
    "coverage_refs": [],
    "chapter_paths": [
        "foundations/deep/policy-gradient/"
    ],
    "description": "一步Actor-Critic对照的实际CPU教学实验；设置、公式和边界详见本文件。",
    "question": "实际学习或状态更新怎样改变此教学任务的独立测量？",
    "sources": [
        "https://spinningup.openai.com/en/latest/algorithms/vpg.html"
    ],
    "defaults": {
        "steps": 600
    },
    "budget": "environment_steps",
    "budget_note": "按实际环境转移计步；各算法更新/回放频率见本文件循环，环境步相同不等于相同计算成本。",
    "limitations": "小型教学任务，不是原论文性能复现；组件实验不计作完整原方法。冻结评价与训练更新分离，预算不代表相同计算成本。"
}


def update(actor,critic,actor_opt,value_opt,x,actions,adv,returns):
    # 步骤1：固定优势为 score function 梯度权重，不对回报求导。
    logp=torch.distributions.Categorical(logits=actor(x)).log_prob(actions)
    loss=-(logp*adv.detach()).mean()
    actor_opt.zero_grad(); loss.backward(); actor_opt.step()
    value_loss=(critic(x).squeeze(-1)-returns.detach()).square().mean()
    value_opt.zero_grad(); value_loss.backward(); value_opt.step()
    return float(loss.detach()),float(value_loss.detach())

def run(seed=0,steps=600,emit=None):
    setup(seed,steps)
    actor,critic=mlp(6,2),mlp(6,1)
    actor_opt=torch.optim.Adam(actor.parameters(),lr=.003)
    value_opt=torch.optim.Adam(critic.parameters(),lr=.01)
    env=Chain(); x=env.reset(); rows=[]; ticks=grid(steps)
    xs=[]; actions=[]; rewards=[]; values=[]; next_values=[]; terminals=[]
    count=0; loss=vloss=0.
    log(rows,0,chain_score(actor),emit)
    for t in range(1,steps+1):
        # 步骤2：先用旧策略交互，先保存下一状态价值，再做任何更新。
        with torch.no_grad():
            dist=torch.distributions.Categorical(logits=actor(x)); a=dist.sample()
            v=float(critic(x).item())
        xp,r,d=env.step(int(a))
        with torch.no_grad(): nv=float(critic(xp).item())
        xs.append(x); actions.append(a); rewards.append(r)
        values.append(v); next_values.append(nv); terminals.append(d)
        x=env.reset() if d else xp
        if t in ticks:
            returns=torch.tensor(rewards)+(1-torch.tensor(terminals,dtype=torch.float32))*torch.tensor(next_values)
            adv=returns-torch.tensor(values)
            loss,vloss=update(actor,critic,actor_opt,value_opt,torch.stack(xs),torch.stack(actions),adv,returns)
            count+=1
            xs=[]; actions=[]; rewards=[]; values=[]; next_values=[]; terminals=[]
        if t in ticks:
            log(rows,t,chain_score(actor),emit,policy_loss=loss,value_loss=vloss,
                updated_batches=count,pending_samples=len(xs))
    return rows

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META,run)
