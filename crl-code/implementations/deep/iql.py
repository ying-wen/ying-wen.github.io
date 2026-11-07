"""IQL：固定离线离散行为日志上的算法特例。
512 条行为转移完全固定，训练 seed 只改变初始化与重采样。
steps 指训练批次数，日志 budget=training_batches；不冒充在线交互。
每个训练批次执行 V、Q、actor 各一次 optimizer.step，共三次优化器调用。
相同批次预算不代表相同优化器调用次数、梯度计算或计算成本。
训练中没有环境交互，环境只用于独立冻结评价（评价步数不记训练预算）。
首先 V 拟合 target Q 的 .7 expectile：正残差权重 .7，负残差 .3。
再 Q 拟合 r+gamma*(1-d)*V(s_next)，不对未观察动作做 max。
最后 actor 用 exp(3*(Q_target-V)) 加权行为克隆，权重截断到 100。
这个 categorical actor 适配离散日志；没有连续 Gaussian actor 的假设。
Q target 用 Polyak；策略不影响训练样本，因此避免离线策略外推动作。
这只是教学控制，不复现 D4RL，不证明现实数据上的离线鲁棒性。
"""
import copy
import torch
try:
    from ._common import *
except ImportError:
    from _common import *
META = {
    "id": "deep-iql",
    "name": "IQL",
    "family": "deep",
    "chapter_paths": [
        "foundations/deep/offline/",
    ],
    "description": "IQL：显式网络更新的 CPU 教学控制实验",
    "question": "有限预算内冻结评估策略回报如何变化？",
    "baseline": "deep-offline_q_learning",
    "metric": "frozen_evaluation_return",
    "unit": "return",
    "higher_better": True,
    "scope": "teaching-control",
    "task": "offline-deadline-chain-fixed512",
    "sources": [
        "https://arxiv.org/abs/2110.06169",
    ],
    "defaults": {
        "steps": 1000,
    },
    "budget": "training_batches",
    "budget_note": "每批重采样32条固定日志转移；每个训练批次执行 V、Q、actor 各一次 optimizer.step，共三次优化器调用。同批次预算不等于同优化器调用数或计算成本；固定数据共512转移，评价交互不计入训练预算。",
    "limitations": "小型教学控制，不复现论文基准；离散评估使用 argmax 策略，连续评估使用确定性 actor 或 tanh(mean)。该冻结评估回报不等于随机行为策略回报或熵正则训练目标，不证明一般性能优势。",
}

def expectile_loss(residual,tau=.7):
    weight=torch.where(residual>0,tau,1-tau)
    return (weight*residual.square()).mean()

def update(q,target,value,actor,q_opt,v_opt,actor_opt,data):
    x,a,r,xp,d=data
    # 步骤 1：价值 expectile 回归只取数据实际动作的目标 Q。
    with torch.no_grad(): observed_target=target(x).gather(1,a.long()[:,None]).squeeze(-1)
    residual=observed_target-value(x).squeeze(-1)
    v_loss=expectile_loss(residual)
    v_opt.zero_grad(); v_loss.backward(); v_opt.step()
    # 步骤 2：使用更新后的 V，冻结 TD 目标；没有 argmax OOD 动作。
    with torch.no_grad(): y=r+.99*(1-d)*value(xp).squeeze(-1)
    prediction=q(x).gather(1,a.long()[:,None]).squeeze(-1)
    q_loss=(prediction-y).square().mean()
    q_opt.zero_grad(); q_loss.backward(); q_opt.step()
    # 步骤 3：固定正权重的行为克隆。actor 不对 advantage 或 Q 反传。
    with torch.no_grad():
        weights=(3*(observed_target-value(x).squeeze(-1))).exp().clamp(max=100.)
    logp=torch.distributions.Categorical(logits=actor(x)).log_prob(a.long())
    actor_loss=-(weights*logp).mean()
    actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()
    # 步骤 4：三种损失计算/优化完，再平滑更新 Q target。
    polyak(q,target)
    return dict(q_loss=float(q_loss.detach()),value_loss=float(v_loss.detach()),
                actor_loss=float(actor_loss.detach()),max_weight=float(weights.max()))

def run(seed=0,steps=1000,emit=None):
    rng=setup(seed); dataset=fixed_offline_data()
    q=mlp(6,2); target=copy.deepcopy(q).requires_grad_(False)
    q_opt=torch.optim.Adam(q.parameters(),lr=.003)
    value,actor=mlp(6,1),mlp(6,2)
    v_opt=torch.optim.Adam(value.parameters(),lr=.003)
    actor_opt=torch.optim.Adam(actor.parameters(),lr=.003)
    rows=[]; ticks=grid(steps)
    record(rows,0,actor,emit,dataset_transitions=512)
    for t in range(1,steps+1):
        # 步骤 5：固定数据重采样，不把回放后的优化次数叫环境交互。
        data=batch(dataset,rng)
        info=update(q,target,value,actor,q_opt,v_opt,actor_opt,data)
        if t in ticks:
            record(rows,t,actor,emit,training_batches=t,optimizer_steps=3*t,dataset_transitions=512,**info)
    return rows

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META,run)
