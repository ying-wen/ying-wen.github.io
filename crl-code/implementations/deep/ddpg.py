"""DDPG：连续动作有界 LQ 的实际回放训练。
步骤：存转移 -> 冻结 Bellman 目标 -> critic 梯度 -> actor 梯度 -> 目标平滑更新。
原始算法公式见 META.sources；默认规模仅为 CPU 教学控制。
actor 输出未约束数值，tanh 后行动范围为 [-1,1]。
critic 目标 r+gamma*(1-d)*Q_target(s_next,mu_target(s_next))。
actor 损失 -Q(s,mu(s))：Q 参数冻结，Q 对动作的梯度仍保留。
训练使用高斯行动探索，冻结评估完全关闭探索噪声。
Polyak tau=.02 表示 target <- .98*target+.02*online。
时间剩余量属于观察；40 步截止是任务定义的真实终止。
优化用 gamma=.99，评估统一用固定初态集合的未折扣总回报。
"""
import copy
from collections import deque
import torch
try:
    from ._common import *
except ImportError:
    from _common import *
META = {
    "id": "deep-ddpg",
    "name": "DDPG",
    "family": "deep",
    "chapter_paths": [
        "foundations/deep/deterministic-control/",
        "algorithms/policy/",
    ],
    "description": "DDPG：显式网络更新的 CPU 教学控制实验",
    "question": "有限预算内冻结评估策略回报如何变化？",
    "baseline": "deep-td3",
    "metric": "frozen_evaluation_return",
    "unit": "return",
    "higher_better": True,
    "scope": "teaching-control",
    "task": "bounded-lq",
    "sources": [
        "https://spinningup.openai.com/en/latest/algorithms/ddpg.html",
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
        ap=target_actor(xp).tanh()
        tail=targets[0](xp,ap)
        y=r+.99*(1-d)*tail
    losses=[]
    # 步骤 2：每个 critic 独立拟合相同的停止梯度目标。
    for q,opt in zip(critics,q_opts):
        loss=(q(x,a)-y).square().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.detach()))
    actor_loss=0.; actor_updated=False
    if True:
        # 步骤 3：冻结 Q 参数，但保留 Q 对动作的导数，传给 actor。
        for q in critics: q.requires_grad_(False)
        loss=-critics[0](x,actor(x).tanh()).mean()
        actor_opt.zero_grad(); loss.backward(); actor_opt.step()
        for q in critics: q.requires_grad_(True)
        actor_loss=float(loss.detach()); actor_updated=True
        # 步骤 4：策略更新后，Polyak 目标跟随当前参数。
        polyak(actor,target_actor)
        for q,target in zip(critics,targets): polyak(q,target)
    return dict(q_loss=sum(losses)/len(losses),actor_loss=actor_loss,actor_updated=int(actor_updated))

def run(seed=0,steps=1000,emit=None):
    rng=setup(seed)
    actor=mlp(2,1)
    target_actor=copy.deepcopy(actor).requires_grad_(False)
    critics=[Q() for _ in range(1)]
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
