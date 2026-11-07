"""TD3：连续动作有界 LQ 的实际回放训练。
步骤：存转移 -> 冻结 Bellman 目标 -> critic 梯度 -> actor 梯度 -> 目标平滑更新。
原始算法公式见 META.sources；默认规模仅为 CPU 教学控制。
相对 DDPG 的三处变化都在 update 中可见：双 critic 的目标最小值、
目标动作平滑噪声、每两次 critic 更新才更新 actor 与所有 targets。
目标噪声先截断到 [-.5,.5]，加到 tanh 动作，再截断到合法动作范围。
actor 只最大化 Q1；双 critic 用于保守目标而不是双策略更新。
训练行动探索和目标动作平滑承担不同角色，评估两种噪声都关闭。
critic 优化之后才做延迟 actor 梯度，最后才 Polyak 更新。
"""
import copy
from collections import deque
import torch
try:
    from ._common import *
except ImportError:
    from _common import *
META = {
    "id": "deep-td3",
    "name": "TD3",
    "family": "deep",
    "chapter_paths": [
        "foundations/deep/deterministic-control/",
        "algorithms/policy/",
    ],
    "description": "TD3：显式网络更新的 CPU 教学控制实验",
    "question": "有限预算内冻结评估策略回报如何变化？",
    "baseline": "deep-ddpg",
    "metric": "frozen_evaluation_return",
    "unit": "return",
    "higher_better": True,
    "scope": "teaching-control",
    "task": "bounded-lq",
    "sources": [
        "https://spinningup.openai.com/en/latest/algorithms/td3.html",
    ],
    "defaults": {
        "steps": 1000,
    },
    "budget": "environment_steps",
    "limitations": "小型教学控制，不复现论文基准；离散评估使用 argmax 策略，连续评估使用确定性 actor 或 tanh(mean)。该冻结评估回报不等于随机行为策略回报或熵正则训练目标，不证明一般性能优势。",
}

def update(actor,critics,target_actor,targets,actor_opt,q_opts,data,iteration):
    x,a,r,xp,d=data
    with torch.no_grad():
        # 步骤 1：先冻结目标，所有目标网络参数均不接受优化器更新。
        # TD3 target policy smoothing：只给目标动作加截断噪声。
        noise=(.2*torch.randn_like(a)).clamp(-.5,.5)
        ap=(target_actor(xp).tanh()+noise).clamp(-1,1)
        tail=torch.minimum(targets[0](xp,ap),targets[1](xp,ap))
        y=r+.99*(1-d)*tail
    losses=[]
    # 步骤 2：每个 critic 独立拟合相同的停止梯度目标。
    for q,opt in zip(critics,q_opts):
        loss=(q(x,a)-y).square().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
    actor_loss=0.; actor_updated=False
    if iteration%2==0:
        # 步骤 3：冻结 Q 参数，但保留 Q 对动作的导数，传给 actor。
        for q in critics: q.requires_grad_(False)
        loss=-critics[0](x,actor(x).tanh()).mean()
        actor_opt.zero_grad(); loss.backward(); actor_opt.step()
        for q in critics: q.requires_grad_(True)
        actor_loss=float(loss.detach()); actor_updated=True
        # 步骤 4：TD3 延迟策略和全部目标更新，与 critic 更新频率区别明确。
        polyak(actor,target_actor)
        for q,target in zip(critics,targets): polyak(q,target)
    return dict(q_loss=sum(losses)/len(losses),actor_loss=actor_loss,actor_updated=int(actor_updated))

def run(seed=0,steps=1000,emit=None):
    rng=setup(seed)
    actor=mlp(2,1)
    target_actor=copy.deepcopy(actor).requires_grad_(False)
    critics=[Q() for _ in range(2)]
    targets=[copy.deepcopy(q).requires_grad_(False) for q in critics]
    actor_opt=torch.optim.Adam(actor.parameters(),lr=.001)
    q_opts=[torch.optim.Adam(q.parameters(),lr=.002) for q in critics]
    env=BoundedLQ(seed); x=env.reset(); replay=deque(maxlen=4000)
    rows=[]; ticks=grid(steps); iteration=0; diagnostic={}; actor_updates=0
    record(rows,0,actor,emit,continuous=True,stochastic=False)
    for t in range(1,steps+1):
        # 步骤 5：前 64 步均匀探索；之后使用当前策略探索动作。
        with torch.no_grad():
            if t<=64: a=torch.tensor([rng.uniform(-1,1)])
            else:
                a=(actor(x).tanh()+torch.tensor([rng.gauss(0,.15)])).clamp(-1,1)
        xp,r,d=env.step(a)
        replay.append((x,a,r,xp,d)); x=env.reset() if d else xp
        if len(replay)>=32:
            iteration+=1
            diagnostic=update(actor,critics,target_actor,targets,actor_opt,q_opts,
                              batch(replay,rng),iteration)
            actor_updates+=diagnostic["actor_updated"]
        if t in ticks:
            record(rows,t,actor,emit,continuous=True,stochastic=False,
                   samples=t,replay_size=len(replay),critic_updates=iteration,
                   actor_updates=actor_updates,target_updates=actor_updates,**diagnostic)
    return rows

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META,run)
