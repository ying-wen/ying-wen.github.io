"""PPO：离散动作 on-policy 教学实现。
编号步骤对应采样、估计、策略更新、价值回归。小任务结果不代表基准性能。
PPO-Clip 使用比值 exp(log pi_new-log pi_old)，旧概率固定不重算。
目标 min(ratio*A,clip(ratio,.8,1.2)*A) 对两种优势符号均成立。
GAE 使用 lambda=.95；真实终止截断递推，批次截止 bootstrap。
同批数据最多四次策略/价值更新，随后清空，不能进入经验回放。
样本近似 KL=(ratio-1)-log(ratio) 超过 .03 时提前停止策略更新。
clip 是移除过大变动的收益激励，不是策略变化的硬约束。
本实现使用全批次；没有复现 MPI、大规模 minibatch 或标准任务成绩。
来源：https://spinningup.openai.com/en/latest/algorithms/ppo.html
"""
import torch
try:
    from ._common import *
except ImportError:
    from _common import *
META = {
    "id": "deep-ppo",
    "name": "PPO",
    "family": "deep",
    "chapter_paths": [
        "foundations/deep/trust-region/",
        "algorithms/policy/",
    ],
    "description": "PPO：显式网络更新的 CPU 教学控制实验",
    "question": "有限预算内冻结评估策略回报如何变化？",
    "baseline": "deep-vpg",
    "metric": "frozen_evaluation_return",
    "unit": "return",
    "higher_better": True,
    "scope": "teaching-control",
    "task": "deadline-chain",
    "sources": [
        "https://spinningup.openai.com/en/latest/algorithms/ppo.html",
    ],
    "defaults": {
        "steps": 1000,
    },
    "budget": "environment_steps",
    "limitations": "小型教学控制，不复现论文基准；离散评估使用 argmax 策略，连续评估使用确定性 actor 或 tanh(mean)。该冻结评估回报不等于随机行为策略回报或熵正则训练目标，不证明一般性能优势。",
}


def update(actor,critic,actor_opt,value_opt,x,a,oldlogp,adv,returns):

    # 步骤 3：旧 logp、优势、回报在多轮优化中保持冻结。
    adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    policy_loss=0.; kl=0.
    for _ in range(4):
        logp=torch.distributions.Categorical(logits=actor(x)).log_prob(a)
        ratio=(logp-oldlogp).exp()
        kl=float((ratio-1-(logp-oldlogp)).mean().detach())
        if kl>.03: break
        surrogate=torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv)
        loss=-surrogate.mean()
        actor_opt.zero_grad(); loss.backward(); actor_opt.step()
        policy_loss=float(loss.detach())
    for _ in range(4):
        value_loss=(critic(x).squeeze(-1)-returns).square().mean()
        value_opt.zero_grad(); value_loss.backward(); value_opt.step()
    return policy_loss,float(value_loss.detach())

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

            # GAE 仅跨未终止转移传递；采样截止使用 V(s_next) bootstrap。
            adv,returns=advantages(rs,vs,nvs,ds,lam=.95)

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
