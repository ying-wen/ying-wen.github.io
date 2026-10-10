"""A2C：离散动作 on-policy 教学实现。
编号步骤对应采样、估计、策略更新、价值回归。小任务结果不代表基准性能。
这是单 worker、同步的一步 advantage actor-critic（A2C 教学特例）。
没有 A3C 的异步线程，也没有多环境并行吞吐优势。
目标 y=r+(1-d)V(s_next)，优势 y-V(s)；两者均在更新前冻结。
真实终止关闭 bootstrap；采样批次截止不会 reset 环境或设 d=True。
先根据旧价值形成 TD 权重，再更新 actor 和 critic，数据只用一次。
gamma=1 与 DeadlineChain 的有限时域总回报一致。
来源：https://spinningup.openai.com/en/latest/algorithms/vpg.html
"""
import torch
try:
    from ._common import *
except ImportError:
    from _common import *
META = {
    "id": "deep-a2c",
    "name": "A2C",
    "family": "deep",
    "chapter_paths": [
        "foundations/deep/policy-gradient/",
        "algorithms/policy/",
    ],
    "description": "A2C：显式网络更新的 CPU 教学控制实验",
    "question": "有限预算内冻结评估策略回报如何变化？",
    "baseline": "deep-vpg",
    "metric": "frozen_evaluation_return",
    "unit": "return",
    "higher_better": True,
    "scope": "teaching-control",
    "task": "deadline-chain",
    "sources": [
        "https://arxiv.org/abs/1602.01783",
    ],
    "defaults": {
        "steps": 1000,
    },
    "budget": "environment_steps",
    "limitations": "小型教学控制，不复现论文基准；离散评估使用 argmax 策略，连续评估使用确定性 actor 或 tanh(mean)。该冻结评估回报不等于随机行为策略回报或熵正则训练目标，不证明一般性能优势。",
}

def update(actor,critic,actor_opt,value_opt,x,a,oldlogp,adv,returns):

    # 步骤 3：优势必须 detach，策略梯度不会反向穿过价值目标。
    dist=torch.distributions.Categorical(logits=actor(x))
    loss=-(dist.log_prob(a)*adv.detach()).mean()
    actor_opt.zero_grad(); loss.backward(); actor_opt.step()
    value_loss=(critic(x).squeeze(-1)-returns.detach()).square().mean()
    value_opt.zero_grad(); value_loss.backward(); value_opt.step()
    return float(loss.detach()),float(value_loss.detach())

def run(seed=0,steps=1000,emit=None):
    setup(seed); actor,critic=mlp(6,2),mlp(6,1)
    actor_opt=torch.optim.Adam(actor.parameters(),lr=.003)
    value_opt=torch.optim.Adam(critic.parameters(),lr=.01)
    env=DeadlineChain(); x=env.reset(); rows=[]; ticks=grid(steps)
    xs=[]; acts=[]; lps=[]; rs=[]; vs=[]; nvs=[]; ds=[]
    record(rows,0,actor,emit); pl=vl=0.; batches=0
    for t in range(1,steps+1):
        # 步骤 1：同一策略采样；缓存旧 logp 和旧价值，不能更新后重算。
        with torch.no_grad():
            dist=torch.distributions.Categorical(logits=actor(x))
            a=dist.sample(); lp=dist.log_prob(a); v=float(critic(x).item())
        xp,r,d=env.step(int(a))
        with torch.no_grad(): nv=float(critic(xp).item())
        xs.append(x); acts.append(a); lps.append(lp)
        rs.append(r); vs.append(v); nvs.append(nv); ds.append(d)
        x=env.reset() if d else xp
        # 步骤 2：每个评估网格前用新采样批次更新；截止不重置环境。
        if t in ticks:

            # 一步 TD actor-critic；有限任务 gamma=1，与总回报目标一致。
            returns=torch.tensor(rs)+(1-torch.tensor(ds,dtype=torch.float32))*torch.tensor(nvs)
            adv=returns-torch.tensor(vs)

            pl,vl=update(actor,critic,actor_opt,value_opt,torch.stack(xs),
                         torch.stack(acts),torch.stack(lps),adv,returns)
            batches+=1
            xs=[]; acts=[]; lps=[]; rs=[]; vs=[]; nvs=[]; ds=[]
        if t in ticks:
            record(rows,t,actor,emit,policy_loss=pl,value_loss=vl,
                   samples=t,updated_batches=batches,pending_samples=len(xs))
    return rows

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META,run)
