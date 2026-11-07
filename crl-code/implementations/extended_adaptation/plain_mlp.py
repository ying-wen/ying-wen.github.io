"""切换回归上的神经元替换组件。每步先反向更新，再统计激活、决定替换。
SGD 无动量故没有优化器矩状态；替换入边重新随机初始化，出边清零以降低扰动。
ReDo 使用相对平均激活阈值；CBP 使用贡献效用 EMA/偏差校正、成熟期和累计替换信用。
这是单隐层回归机制实验，不是原文 Atari/持续强化学习完整实验。
主指标冻结当前函数 MSE，旧函数 MSE 是诊断；第二阶段目标本身相冲突，旧误差
不能单独解释为算法缺陷或声称保持全部旧知识。预算为训练样本，不是 optimizer 次数。
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
    "id": "extended-plain_mlp",
    "name": "不替换 MLP SGD",
    "family": "extended_adaptation",
    "task": "switch-regression",
    "baseline": "extended-redo",
    "metric": "冻结当前函数均方误差",
    "unit": "squared_error",
    "higher_better": False,
    "scope": "teaching-regression",
    "implementation_kind": "teaching_method",
    "coverage_refs": [],
    "chapter_paths": [
        "algorithms/meta/"
    ],
    "description": "不替换 MLP SGD的实际CPU教学实验；设置、公式和边界详见本文件。",
    "question": "实际学习或状态更新怎样改变此教学任务的独立测量？",
    "sources": [
        "https://arxiv.org/abs/2302.12902"
    ],
    "defaults": {
        "steps": 600
    },
    "budget": "training_samples",
    "budget_note": "每样本一次监督SGD；回收检查/效用更新不增加SGD次数。",
    "limitations": "小型教学任务，不是原论文性能复现；组件实验不计作完整原方法。冻结评价与训练更新分离，预算不代表相同计算成本。"
}

def replace_units(net,indices,rng):
    # 与 nn.Linear 默认初始化一致：入边范围 ±1/sqrt(fan_in)。
    with torch.no_grad():
        for j in indices:
            net.hidden.weight[j].copy_(torch.tensor([rng.uniform(-.5,.5) for _ in range(4)]))
            net.hidden.bias[j]=0.
            net.output.weight[:,j]=0.

def run(seed=0,steps=600,emit=None):
    rng=setup(seed,steps); reset_rng=random.Random(seed+822)
    net=ReLUNet(); opt=torch.optim.SGD(net.parameters(),lr=.03)
    util=torch.zeros(16); age=torch.zeros(16); credit=0.; buffer=[]
    rows=[]; points=grid(steps); replaced=0
    def evaluate(t):
        current,old=regression_scores(net,t>steps//2)
        log(rows,t,current,emit,old_mse=old,replaced_units=replaced,
            optimizer_steps=t,training_samples=t)
    evaluate(0)
    for t in range(1,steps+1):
        x,y=regression_data(rng,t>steps//2)
        opt.zero_grad(); loss=.5*(net(x)-y).square(); loss.backward(); opt.step()
        with torch.no_grad():
            features=net.hidden(x).relu()
            buffer.append(x)
            if len(buffer)>32:buffer.pop(0)
            age+=1
            util=.99*util+.01*features.abs()*net.output.weight.abs().mean(0)
            selected=[]
            pass
            if selected:
                replace_units(net,selected,reset_rng); replaced+=len(selected)
                util[selected]=0.; age[selected]=0.
        if t in points:evaluate(t)
    return rows

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META,run)
