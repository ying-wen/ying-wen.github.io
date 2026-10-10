"""C51 categorical 回报分布学习：不只拟合回报均值。
来源见 META；离散 DeadlineChain 对照 DQN，统一冻结贪心评价。
固定 51 个原子支持 [-1,1]，对应本任务可能回报的保守覆盖。
Bellman 操作移动目标原子后，线性投影把概率质量送回固定支持。
整数投影索引相等时，质量必须完整落在该原子，不能被两个零权重丢弃。
动作选择仍使用分布期望，交叉熵目标不反向穿过目标网络。
教学规模不复现 Atari 成绩；目标使用 gamma=.99，评价记录未折扣回报。
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
    "id": "deep-c51",
    "name": "C51",
    "family": "deep",
    "chapter_paths": [
        "foundations/deep/distributional/",
        "algorithms/value/",
    ],
    "description": "C51：显式网络更新的 CPU 教学控制实验",
    "question": "有限预算内冻结评估策略回报如何变化？",
    "baseline": "deep-dqn",
    "metric": "frozen_evaluation_return",
    "unit": "return",
    "higher_better": True,
    "scope": "teaching-control",
    "task": "deadline-chain",
    "sources": [
        "https://arxiv.org/abs/1707.06887",
    ],
    "defaults": {
        "steps": 1000,
    },
    "budget": "environment_steps",
    "limitations": "小型教学控制，不复现论文基准；离散评估使用 argmax 策略，连续评估使用确定性 actor 或 tanh(mean)。该冻结评估回报不等于随机行为策略回报或熵正则训练目标，不证明一般性能优势。",
}

class DistributionQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.n=51; self.net=mlp(6,2*self.n)
        self.register_buffer("support",torch.linspace(-1,1,self.n))
    def distribution(self,x):
        values=self.net(x).reshape(*x.shape[:-1],2,self.n)
        return values.softmax(-1)
    def forward(self,x):
        values=self.distribution(x)
        return (values*self.support).sum(-1)

def project(probability,rewards,terminals,support):
    """将移动原子的质量投回支持，包含恰落在原子上的边界情形。"""
    n=len(support); dz=(support[-1]-support[0])/(n-1)
    transformed=(rewards[:,None]+.99*(1-terminals[:,None])*support).clamp(support[0],support[-1])
    position=(transformed-support[0])/dz
    lower=position.floor().long(); upper=position.ceil().long()
    result=torch.zeros_like(probability)
    # lower==upper 时完整质量给 lower，否则按距离给相邻两个原子。
    result.scatter_add_(1,lower,probability*(upper-position+(lower==upper).float()))
    result.scatter_add_(1,upper,probability*(position-lower))
    return result

def update(online,target,opt,data):
    x,a,r,xp,d=data
    with torch.no_grad():
        # 步骤 1：目标网络按期望选动作，再做分布 Bellman 投影。
        selected=target(xp).argmax(-1)
        p=target.distribution(xp)[torch.arange(len(x)),selected]
        projected=project(p,r,d,online.support)
    # 步骤 2：交叉熵梯度只流入执行动作对应的在线 logits。
    logits=online.net(x).reshape(-1,2,online.n)[torch.arange(len(x)),a.long()]
    loss=-(projected*logits.log_softmax(-1)).sum(-1).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    return float(loss.detach())

def run(seed=0,steps=1000,emit=None):
    rng=setup(seed); online=DistributionQ()
    target=copy.deepcopy(online).requires_grad_(False)
    opt=torch.optim.Adam(online.parameters(),lr=.003)
    env=DeadlineChain(); x=env.reset(); replay=deque(maxlen=4000)
    rows=[]; ticks=grid(steps); loss=0.; updates=0
    record(rows,0,online,emit)
    for t in range(1,steps+1):
        # 步骤 3：行为策略用分布期望的 epsilon-greedy，先存真实转移。
        eps=max(.05,1-t/max(1,steps*.7))
        with torch.no_grad():
            a=rng.randrange(2) if rng.random()<eps else int(online(x).argmax())
        xp,r,d=env.step(a)
        replay.append((x,torch.tensor(a),r,xp,d)); x=env.reset() if d else xp
        if len(replay)>=32 and t%2==0:
            loss=update(online,target,opt,batch(replay,rng)); updates+=1
        # 步骤 4：同 DQN 的硬更新时序；没有提前使用未来参数。
        if t%100==0: target.load_state_dict(online.state_dict())
        if t in ticks: record(rows,t,online,emit,loss=loss,gradient_updates=updates,samples=t)
    return rows

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META,run)
