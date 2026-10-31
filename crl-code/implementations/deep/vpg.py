"""VPG：离散动作 on-policy 教学实现。
编号步骤对应采样、估计、策略更新、价值回归。小任务结果不代表基准性能。
这是 REINFORCE reward-to-go 加学习价值基线的 VPG 版本。
G_t=sum_{k=t}^{T-1} r_k；优势 G_t-V(s_t) 只作为固定梯度权重。
每个完整回合更新一次，随后旧数据清空，不能再当成 on-policy 数据。
预算结束时未完成回合不用于 Monte Carlo 更新，但仍计入交互预算。
critic 最小化 (V-G)^2，actor 最小化 -log pi(a|s)*(G-V)。
gamma=1 对应这个有限时域任务的未折扣回报；没有优势归一化。
来源：https://spinningup.openai.com/en/latest/algorithms/vpg.html
"""
import torch
try:
    from ._common import *
except ImportError:
    from _common import *
META = {
    "id": "deep-vpg",
    "name": "VPG",
    "family": "deep",
    "chapter_paths": [
        "foundations/deep/policy-gradient/",
        "algorithms/policy/",
    ],
    "description": "VPG：显式网络更新的 CPU 教学控制实验",
    "question": "有限预算内冻结评估策略回报如何变化？",
    "baseline": "deep-a2c",
    "metric": "frozen_evaluation_return",
    "unit": "return",
    "higher_better": True,
    "scope": "teaching-control",
    "task": "deadline-chain",
    "sources": [
        "https://spinningup.openai.com/en/latest/algorithms/vpg.html",
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
        # 步骤 2：VPG 等待真实回合结束，预算末未完成回合舍弃。
        if d:

            # 完整回合 Monte Carlo reward-to-go；不把预算截断冒充零尾回报。
            returns=[]; g=0.
            for reward in reversed(rs): g=reward+g; returns.append(g)
            returns=torch.tensor(list(reversed(returns)))
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
