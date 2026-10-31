"""QR-DQN quantile 回报分布学习：不只拟合回报均值。
来源见 META；离散 DeadlineChain 对照 DQN，统一冻结贪心评价。
固定 16 个等概率分位点 tau_i=(i+.5)/16，不固定分位值。
两两残差 delta_ij=y_j-theta_i 对应 quantile Huber 回归。
权重 |tau_i-I(delta_ij<0)| 使用预测分位点索引 i，不能用目标索引 j。
动作选择使用分位值平均；目标分位值停止梯度。
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
    "id": "deep-qr_dqn",
    "name": "QR-DQN",
    "family": "deep",
    "chapter_paths": [
        "foundations/deep/distributional/",
        "algorithms/value/",
    ],
    "description": "QR-DQN：显式网络更新的 CPU 教学控制实验",
    "question": "有限预算内冻结评估策略回报如何变化？",
    "baseline": "deep-dqn",
    "metric": "frozen_evaluation_return",
    "unit": "return",
    "higher_better": True,
    "scope": "teaching-control",
    "task": "deadline-chain",
    "sources": [
        "https://arxiv.org/abs/1710.10044",
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
        self.n=16; self.net=mlp(6,2*self.n)
        self.register_buffer("tau",(torch.arange(self.n)+.5)/self.n)
    def distribution(self,x):
        values=self.net(x).reshape(*x.shape[:-1],2,self.n)
        return values
    def forward(self,x):
        values=self.distribution(x)
        return values.mean(-1)

def quantile_loss(prediction,target,tau):
    """预测 i 与目标 j 两两比较；tau 权重广播在预测轴上。"""
    residual=target[:,None,:]-prediction[:,:,None]
    absolute=residual.abs()
    huber=torch.where(absolute<=1,.5*residual.square(),absolute-.5)
    weight=(tau[None,:,None]-(residual.detach()<0).float()).abs()
    return (weight*huber).mean()

def update(online,target,opt,data):
    x,a,r,xp,d=data
    with torch.no_grad():
        # 步骤 1：选取下一动作的所有目标分位值，终止时退化到 reward。
        selected=target(xp).argmax(-1)
        next_quantiles=target.distribution(xp)[torch.arange(len(x)),selected]
        y=r[:,None]+.99*(1-d[:,None])*next_quantiles
    # 步骤 2：无需排序；每个 tau 对应一个回归输出和非对称损失。
    prediction=online.distribution(x)[torch.arange(len(x)),a.long()]
    loss=quantile_loss(prediction,y,online.tau)
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
