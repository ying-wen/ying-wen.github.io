"""DQN：回放、半梯度 TD、冻结目标网络，逐步对齐采样时序。
原始来源：Mnih et al. 2015。评估为独立冻结贪心策略。
阅读顺序：run 中观察转移入库，update 中找到 Bellman 目标和半梯度。
Q 网络输出两个动作值；gather 把 [batch,2] 变成 [batch]，防止广播错误。
学习目标使用 gamma=.99，图中统一记录未折扣的冻结任务回报。
这两个目标有区别；教学曲线不能解释为对优化目标的直接无偏估计。
目标网络是 online 的延迟副本，不能加入 optimizer 或误用实时梯度。
随机回放降低连续样本相关性；它不让有限数据自动变成独立同分布。
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
    "id": "deep-dqn",
    "name": "DQN",
    "family": "deep",
    "chapter_paths": [
        "foundations/deep/deep-value/",
        "algorithms/value/",
    ],
    "description": "DQN：显式网络更新的 CPU 教学控制实验",
    "question": "有限预算内冻结评估策略回报如何变化？",
    "baseline": "deep-double_dqn",
    "metric": "frozen_evaluation_return",
    "unit": "return",
    "higher_better": True,
    "scope": "teaching-control",
    "task": "deadline-chain",
    "sources": [
        "https://www.nature.com/articles/nature14236",
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
        tail=target(xp).max(-1).values
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
