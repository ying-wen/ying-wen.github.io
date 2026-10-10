"""非线性RTU单旋转块的实际预测训练，Eq(4)+对数r/theta+输入归一化。
r=exp(-exp(nu)), angle=exp(logangle), k=sqrt(1-r²)；
h_t=tanh(r R(angle)h_{t-1}+k W_x x_t)，输出使用两个实通道。
每段用RTRL J_t=D_t*rR*J_{t-1}+局部参数Jacobian，训练nu/angle/Wx/readout。
不是旧固定旋转的标量核；这里所有7个参数实际更新。
参数段内固定，故Jacobian是精确导数；段间更新并清状态/迹。
没有完整PPO/actor-critic，没有论文持续流控制或跨段变参数Jacobian，故明确组件实验。
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
    "id": "extended-rtu",
    "name": "RTU可训练旋转块实验",
    "family": "extended_adaptation",
    "task": "delayed-cue-sequences",
    "baseline": "extended-bptt",
    "metric": "frozen_sequence_mse",
    "unit": "squared_error",
    "higher_better": False,
    "scope": "teaching-prediction",
    "implementation_kind": "component_experiment",
    "coverage_refs": [
        "core-rtu"
    ],
    "chapter_paths": [
        "construction/state/"
    ],
    "description": "RTU可训练旋转块实验的实际CPU教学实验；设置、公式和边界详见本文件。",
    "question": "实际学习或状态更新怎样改变此教学任务的独立测量？",
    "sources": [
        "https://arxiv.org/html/2409.01449v1"
    ],
    "defaults": {
        "steps": 600
    },
    "budget": "sequences",
    "budget_note": "每序列8个输入、一次末端监督和一次参数更新；element_steps另记实际输入数。",
    "limitations": "一对非线性旋转递归单元的精确序列内RTRL；参数在序列内固定、末端更新，不是跨更新无限流精确梯度，也非作者完整PPO系统。"
}

def recurrent(theta,inputs):
    nu,logangle=theta[0],theta[1]; inputs_w=theta[2:4]
    r=torch.exp(-torch.exp(nu)); dr=-torch.exp(nu)*r
    angle=torch.exp(logangle); c,s=angle.cos(),angle.sin()
    rotation=torch.stack([c,-s,s,c]).reshape(2,2)
    derivative_rotation=torch.stack([-s,-c,c,-s]).reshape(2,2)
    scale=torch.sqrt(1-r*r); dscale=-r*dr/scale
    h=torch.zeros(2,dtype=theta.dtype); jac=torch.zeros(2,4,dtype=theta.dtype)
    for x in inputs:
        previous=h
        h=(r*rotation@previous+scale*inputs_w*x).tanh()
        activation=1-h.square()
        # 局部导数与传播项都读取同一旧参数，不构建跨时间自动求导图。
        direct=torch.stack([dr*rotation@previous+dscale*inputs_w*x,
                            r*derivative_rotation@previous*angle,
                            torch.tensor([1.,0.],dtype=theta.dtype)*scale*x,
                            torch.tensor([0.,1.],dtype=theta.dtype)*scale*x],dim=1)
        jac=activation[:,None]*(r*rotation@jac+direct)
    return h,jac

def predict(theta,inputs):
    h,_=recurrent(theta,inputs)
    return theta[4:6]@h+theta[6]

def loss_gradient(theta,inputs,target):
    h,jac=recurrent(theta,inputs)
    error=theta[4:6]@h+theta[6]-target
    gradient=torch.cat([error*(theta[4:6]@jac),error*h,error.reshape(1)])
    return .5*error.square(),gradient

def run(seed=0,steps=600,emit=None):
    rng=setup(seed,steps)
    theta=torch.tensor([math.log(-math.log(.8)),math.log(.2),.3,-.2,.4,-.2,0.])
    rows=[]; ticks=grid(steps); loss=0.
    log(rows,0,sequence_score(lambda xs:predict(theta,xs)),emit)
    for t in range(1,steps+1):
        xs,y=sequence(rng)
        objective,gradient=loss_gradient(theta,xs,y)
        theta=theta-.03*gradient
        if t in ticks:
            log(rows,t,sequence_score(lambda xs:predict(theta,xs)),emit,
                train_loss=float(objective),gradient_norm=float(gradient.norm()),
                radius=float(torch.exp(-torch.exp(theta[0]))),element_steps=8*t)
    return rows

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META,run)
