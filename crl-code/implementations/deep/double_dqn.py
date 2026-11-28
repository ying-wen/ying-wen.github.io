"""Double DQN：回放、半梯度 TD、冻结目标网络，逐步对齐采样时序。
原始来源：van Hasselt et al. 2016。评估为独立冻结贪心策略。
关键区别：online 在 next state 选择动作，target 评价这个动作。
若两个网络的排序不同，DDQN 目标就与 DQN 的 max(target) 不同。
该拆分缓解最大化偏差，不保证每次更新都优于 DQN。
update 的 gather 结果为 [batch]；整个目标在 no_grad 下形成。
学习用 gamma=.99，图记录未折扣任务回报，默认仅做小任务对照。
"""
import copy
from collections import deque
import torch
from torch import nn
try:
    from ._common import *
except ImportError:
    from _common import *
META = {
    "id": "deep-double_dqn",
    "name": "Double DQN",
    "family": "deep",
    "chapter_paths": [
        "foundations/deep/deep-value/",
        "algorithms/value/",
    ],
    "description": "Double DQN：显式网络更新的 CPU 教学控制实验",
    "question": "有限预算内冻结评估策略回报如何变化？",
    "baseline": "deep-dqn",
    "metric": "frozen_evaluation_return",
    "unit": "return",
    "higher_better": True,
    "scope": "teaching-control",
    "task": "deadline-chain",
    "sources": [
        "https://arxiv.org/abs/1509.06461",
    ],
    "defaults": {
        "steps": 1000,
    },
    "budget": "environment_steps",
    "limitations": "小型教学控制，不复现论文基准；离散评估使用 argmax 策略，连续评估使用确定性 actor 或 tanh(mean)。该冻结评估回报不等于随机行为策略回报或熵正则训练目标，不证明一般性能优势。",
}


def update(online,target,opt,data):
    x,a,r,xp,d=data
    # 步骤 1：当前网络只对真实执行动作的 Q 值求导。
    prediction=online(x).gather(1,a.long().reshape(-1,1)).squeeze(-1)
    with torch.no_grad():
        selected=online(xp).argmax(-1)
        tail=target(xp).gather(1,selected[:,None]).squeeze(-1)
        # 步骤 2：真实终止不 bootstrap；目标完全停止梯度。
        y=r+.99*(1-d)*tail
    loss=nn.functional.smooth_l1_loss(prediction,y)
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(),10.)
    opt.step()
    return float(loss.detach())

def run(seed=0,steps=1000,emit=None):
    rng=setup(seed); online=mlp(6,2)
    target=copy.deepcopy(online).requires_grad_(False)
    opt=torch.optim.Adam(online.parameters(),lr=.003)
    replay=deque(maxlen=4000); env=DeadlineChain(); x=env.reset()
    rows=[]; ticks=grid(steps); loss=0.; updates=0; target_updates=0
    record(rows,0,online,emit)
    for t in range(1,steps+1):
        # 步骤 3：epsilon 探索，先交互再存入回放，禁止训练未来样本。
        eps=max(.05,1-t/max(1,steps*.7))
        with torch.no_grad():
            a=rng.randrange(2) if rng.random()<eps else int(online(x).argmax())
        xp,r,d=env.step(a)
        replay.append((x,torch.tensor(a),r,xp,d)); x=env.reset() if d else xp
        if len(replay)>=32 and t%2==0:
            loss=update(online,target,opt,batch(replay,rng)); updates+=1
        # 步骤 4：周期硬拷贝发生在 online 更新之后。
        if t%100==0:
            target.load_state_dict(online.state_dict()); target_updates+=1
        if t in ticks:
            record(rows,t,online,emit,loss=loss,samples=t,
                   replay_size=len(replay),gradient_updates=updates,target_updates=target_updates)
    return rows

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META,run)
