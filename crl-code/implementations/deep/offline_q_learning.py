"""Vanilla offline Q-learning：固定数据上只做 Bellman 回归的对照。
数据、网络大小、学习率、训练更新网格和冻结评估与 CQL 一致。
关键差异：没有 logsumexp 保守项，下一动作值使用 max Q_target。
这可能外推未充分覆盖动作；小任务成功不证明离线分布外安全。
steps 指训练批次数，日志 budget=training_batches；不冒充在线交互。
每个训练批次执行 Q 一次 optimizer.step，共一次优化器调用。
相同批次预算不代表相同优化器调用次数、梯度计算或计算成本。
"""
import copy
import torch
try:
    from ._common import *
except ImportError:
    from _common import *
META = {
    "id": "deep-offline_q_learning",
    "name": "Offline Q-learning",
    "family": "deep",
    "chapter_paths": [
        "foundations/deep/offline/",
    ],
    "description": "Offline Q-learning：显式网络更新的 CPU 教学控制实验",
    "question": "有限预算内冻结评估策略回报如何变化？",
    "baseline": "deep-cql",
    "metric": "frozen_evaluation_return",
    "unit": "return",
    "higher_better": True,
    "scope": "teaching-control",
    "task": "offline-deadline-chain-fixed512",
    "sources": [
        "https://www.nature.com/articles/nature14236",
    ],
    "defaults": {
        "steps": 1000,
    },
    "budget": "training_batches",
    "budget_note": "每批重采样32条固定日志转移；每个训练批次执行 Q 一次 optimizer.step，共一次优化器调用。同批次预算不等于同优化器调用数或计算成本；固定数据共512转移，评价交互不计入训练预算。",
    "limitations": "小型教学控制，不复现论文基准；离散评估使用 argmax 策略，连续评估使用确定性 actor 或 tanh(mean)。该冻结评估回报不等于随机行为策略回报或熵正则训练目标，不证明一般性能优势。",
}

def update(q,target,opt,data):
    x,a,r,xp,d=data
    # 步骤 1：冻结目标，并由真实终止标记屏蔽下一状态价值。
    with torch.no_grad(): y=r+.99*(1-d)*target(xp).max(-1).values
    prediction=q(x).gather(1,a.long()[:,None]).squeeze(-1)
    # 步骤 2：只有数据实际动作的 TD 回归，没有其他保守或行为克隆项。
    loss=.5*(prediction-y).square().mean()
    opt.zero_grad(); loss.backward(); opt.step()
    # 步骤 3：优化后再更新目标，保持与 CQL 同样的 tau=.02。
    polyak(q,target)
    return float(loss.detach())

def run(seed=0,steps=1000,emit=None):
    rng=setup(seed); dataset=fixed_offline_data()
    q=mlp(6,2); target=copy.deepcopy(q).requires_grad_(False)
    opt=torch.optim.Adam(q.parameters(),lr=.003)
    rows=[]; ticks=grid(steps)
    record(rows,0,q,emit,dataset_transitions=512)
    for t in range(1,steps+1):
        # 步骤 4：离线重采样；冻结评价不会影响数据生成或学习随机流。
        loss=update(q,target,opt,batch(dataset,rng))
        if t in ticks: record(rows,t,q,emit,training_batches=t,optimizer_steps=t,dataset_transitions=512,td_loss=loss)
    return rows

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META,run)
