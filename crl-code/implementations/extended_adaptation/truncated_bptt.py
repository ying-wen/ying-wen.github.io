"""长度8延迟线索的双隐藏单元RNN实际训练，参数段内冻结。
h_t=tanh(W h_{t-1}+b x_t), y_hat=o^T h_8+c；目标记住首位±1。
TBPTT只对最后3步反传，边界保留隐藏数值但detach梯度。
每训练序列一次SGD更新，旧序列图立即释放；独立固定32序列冻结测MSE。
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
    "id": "extended-truncated_bptt",
    "name": "TBPTT三步对照",
    "family": "extended_adaptation",
    "task": "delayed-cue-sequences",
    "baseline": "extended-bptt",
    "metric": "frozen_sequence_mse",
    "unit": "squared_error",
    "higher_better": False,
    "scope": "teaching-prediction",
    "implementation_kind": "teaching_method",
    "coverage_refs": [],
    "chapter_paths": [
        "construction/state/"
    ],
    "description": "TBPTT三步对照的实际CPU教学实验；设置、公式和边界详见本文件。",
    "question": "实际学习或状态更新怎样改变此教学任务的独立测量？",
    "sources": [
        "https://doi.org/10.1162/neco.1990.2.4.490"
    ],
    "defaults": {
        "steps": 600
    },
    "budget": "sequences",
    "budget_note": "每序列8个输入、一次末端监督和一次参数更新；element_steps另记实际输入数。",
    "limitations": "小型教学任务，不是原论文性能复现；组件实验不计作完整原方法。冻结评价与训练更新分离，预算不代表相同计算成本。"
}

def predict(theta,inputs,truncate=False):
    W=theta[:4].reshape(2,2); b=theta[4:6]; out=theta[6:8]; bias=theta[8]
    h=torch.zeros(2,dtype=theta.dtype)
    for t,x in enumerate(inputs):
        if truncate and t==len(inputs)-3: h=h.detach()
        h=(W@h+b*x).tanh()
    return out@h+bias

def loss_gradient(theta,inputs,target,truncate=False):
    prediction=predict(theta,inputs,truncate)
    loss=.5*(prediction-target).square()
    return loss,torch.autograd.grad(loss,theta)[0]

def run(seed=0,steps=600,emit=None):
    rng=setup(seed,steps)
    theta=torch.tensor([.7,.1,-.1,.7,.3,-.2,.4,-.2,0.],requires_grad=True)
    rows=[]; ticks=grid(steps); loss=0.
    log(rows,0,sequence_score(lambda xs:predict(theta,xs)),emit)
    for t in range(1,steps+1):
        xs,y=sequence(rng)
        objective,gradient=loss_gradient(theta,xs,y,truncate=True)
        # 步骤：本段完整梯度先算完，再原地修改参数；下一段从h=0开始。
        with torch.no_grad():theta-=.03*gradient
        loss=float(objective.detach())
        if t in ticks:
            log(rows,t,sequence_score(lambda xs:predict(theta,xs)),emit,
                train_loss=loss,gradient_norm=float(gradient.norm()),element_steps=8*t)
    return rows

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META,run)
