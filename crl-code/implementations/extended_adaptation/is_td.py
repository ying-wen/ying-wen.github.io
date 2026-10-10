"""固定目标策略π(1)=.8，行为μ(1)=.5，单状态继续任务γ=.9。
每步动作奖励为action*amplitude，幅度中点1->.5；真实目标值为8或4。
独立采样行为流，学习器不访问解析真值。rho截断上限2大于最大比率1.6，
因此目标不因rho截断改变；c上限1仅控制轨迹校正传播。
IS一步TD仅作同目标、同采样流对照。
"""
import math
import random
import torch
from torch import nn
try:
    from ._common import *
except ImportError:
    from _common import *
META = {
    "id": "extended-is_td",
    "name": "一步IS TD对照",
    "family": "extended_adaptation",
    "task": "offpolicy-one-state",
    "baseline": "extended-vtrace",
    "metric": "frozen_target_value_mse",
    "unit": "squared_error",
    "higher_better": False,
    "scope": "teaching-prediction",
    "implementation_kind": "component_experiment",
    "coverage_refs": [],
    "chapter_paths": [
        "algorithms/credit-assignment/"
    ],
    "description": "一步IS TD对照的实际CPU教学实验；设置、公式和边界详见本文件。",
    "question": "实际学习或状态更新怎样改变此教学任务的独立测量？",
    "sources": [
        "https://arxiv.org/abs/1802.01561"
    ],
    "defaults": {
        "steps": 600
    },
    "budget": "environment_steps",
    "budget_note": "按实际环境转移计步；各算法更新/回放频率见本文件循环，环境步相同不等于相同计算成本。",
    "limitations": "小型教学任务，不是原论文性能复现；组件实验不计作完整原方法。冻结评价与训练更新分离，预算不代表相同计算成本。"
}


def run(seed=0,steps=600,emit=None):
    rng=setup(seed,steps); value=torch.tensor(0.,requires_grad=True)
    opt=torch.optim.SGD([value],lr=.08); rows=[]; ticks=grid(steps)
    rewards=[]; ratios=[]; loss=0.; batches=0
    log(rows,0,64.,emit)
    for t in range(1,steps+1):
        amplitude=1. if t<=steps//2 else .5
        a=int(rng.random()<.5)
        rewards.append(amplitude*a); ratios.append((.8 if a else .2)/.5)
        # 在切换点也结束批次，不能混同两段目标分布。
        if t in ticks or t==steps//2:
            old=float(value.detach()); n=len(rewards)
            target=torch.tensor([old+rho*(r+.9*old-old) for r,rho in zip(rewards,ratios)])
            loss_tensor=.5*(value-target.detach()).square().mean()
            opt.zero_grad(); loss_tensor.backward(); opt.step()
            loss=float(loss_tensor.detach()); batches+=1
            rewards=[]; ratios=[]
        if t in ticks:
            log(rows,t,(float(value.detach())-8*amplitude)**2,emit,
                phase="before_change" if t<=steps//2 else "after_change",
                learned_value=float(value.detach()),loss=loss,training_batches=batches)
    return rows

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META,run)
