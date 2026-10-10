"""CQL：固定离线离散行为日志上的算法特例。
512 条行为转移完全固定，训练 seed 只改变初始化与重采样。
steps 指训练批次数，日志 budget=training_batches；不冒充在线交互。
每个训练批次执行 Q 一次 optimizer.step，共一次优化器调用。
相同批次预算不代表相同优化器调用次数、梯度计算或计算成本。
训练中没有环境交互，环境只用于独立冻结评价（评价步数不记训练预算）。
保守项 alpha*(logsumexp_a Q(s,a)-Q(s,a_data))，这里 alpha=1。
离散动作可以精确求和，不需要连续 CQL 的动作重要性采样近似。
TD 与保守项共同优化；目标 Q 冻结，行为选择不取决于 learner。
这是离散 CQL(H) 固定惩罚版本，没有原论文的 Lagrange 自动权重。
这只是教学控制，不复现 D4RL，不证明现实数据上的离线鲁棒性。
"""
import copy
import torch
try:
    from ._common import *
except ImportError:
    from _common import *
META = {
    "id": "deep-cql",
    "name": "CQL",
    "family": "deep",
    "chapter_paths": [
        "foundations/deep/offline/",
    ],
    "description": "CQL：显式网络更新的 CPU 教学控制实验",
    "question": "有限预算内冻结评估策略回报如何变化？",
    "baseline": "deep-offline_q_learning",
    "metric": "frozen_evaluation_return",
    "unit": "return",
    "higher_better": True,
    "scope": "teaching-control",
    "task": "offline-deadline-chain-fixed512",
    "sources": [
        "https://arxiv.org/abs/2006.04779",
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
    # 步骤 1：冻结 Bellman 目标。真正终止关闭所有尾价值。
    with torch.no_grad(): y=r+.99*(1-d)*target(xp).max(-1).values
    values=q(x); observed=values.gather(1,a.long()[:,None]).squeeze(-1)
    td=.5*(observed-y).square().mean()
    # 步骤 2：拉低整体动作值相对数据动作值，softmax 梯度精确可算。
    penalty=(values.logsumexp(-1)-observed).mean()
    loss=td+penalty
    opt.zero_grad(); loss.backward(); opt.step()
    polyak(q,target)
    return dict(td_loss=float(td.detach()),conservative_penalty=float(penalty.detach()))

def run(seed=0,steps=1000,emit=None):
    rng=setup(seed); dataset=fixed_offline_data()
    q=mlp(6,2); target=copy.deepcopy(q).requires_grad_(False)
    q_opt=torch.optim.Adam(q.parameters(),lr=.003)
    actor=q
    rows=[]; ticks=grid(steps)
    record(rows,0,actor,emit,dataset_transitions=512)
    for t in range(1,steps+1):
        # 步骤 5：固定数据重采样，不把回放后的优化次数叫环境交互。
        data=batch(dataset,rng)
        info=update(q,target,q_opt,data)
        if t in ticks:
            record(rows,t,actor,emit,training_batches=t,optimizer_steps=1*t,dataset_transitions=512,**info)
    return rows

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META,run)
