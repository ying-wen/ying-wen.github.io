"""3维线性任务分布的实际MAML监督meta-training，并测新任务一步适应。
每任务y=x^T w*，w*在[.5,-.3,.8]附近变化；support/query各8独立样本。
内层theta'=theta-.2 grad L_support；外层优化mean L_query(theta')，
完整二阶导数穿过内层梯度。不是scalar二次核、也不是MAML-RL控制复现。
对照直接联合训练query回归，不通过support；评价仍给同样一步support适应。
固定24未参与训练的任务测适应后query MSE；比较体现初始化与适应，不能泛化真实RL。
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
    "id": "extended-joint_training",
    "name": "联合训练初始化对照",
    "family": "extended_adaptation",
    "task": "linear-task-meta-learning",
    "baseline": "extended-maml",
    "metric": "heldout_adapted_query_mse",
    "unit": "squared_error",
    "higher_better": False,
    "scope": "supervised-meta-learning",
    "implementation_kind": "component_experiment",
    "coverage_refs": [],
    "chapter_paths": [
        "algorithms/meta/"
    ],
    "description": "联合训练初始化对照的实际CPU教学实验；设置、公式和边界详见本文件。",
    "question": "实际学习或状态更新怎样改变此教学任务的独立测量？",
    "sources": [
        "https://arxiv.org/abs/1703.03400"
    ],
    "defaults": {
        "steps": 600
    },
    "budget": "meta_training_batches",
    "budget_note": "每批4个任务各8支持+8查询样本；MAML每任务一次可微内层适应、一次共享外层SGD；joint仅共享SGD。",
    "limitations": "小型教学任务，不是原论文性能复现；组件实验不计作完整原方法。冻结评价与训练更新分离，预算不代表相同计算成本。"
}

def task(generator):
    teacher=torch.tensor([.5,-.3,.8])+torch.randn(3,generator=generator)*.5
    support=torch.randn(8,3,generator=generator)
    query=torch.randn(8,3,generator=generator)
    return support,support@teacher,query,query@teacher

def objective(theta,support,sy,query,qy,adapt=True):
    adapted=theta
    if adapt:
        inner=.5*(support@theta-sy).square().mean()
        gradient=torch.autograd.grad(inner,theta,create_graph=True)[0]
        adapted=theta-.2*gradient
    return .5*(query@adapted-qy).square().mean()

def evaluate_initialization(theta):
    generator=torch.Generator().manual_seed(771)
    losses=[]
    # 评价需要support梯度，但不修改训练theta/optimizer/RNG。
    for _ in range(24):
        detached=theta.detach().requires_grad_()
        data=task(generator)
        losses.append(float(2*objective(detached,*data).detach()))
    return sum(losses)/len(losses)

def run(seed=0,steps=600,emit=None):
    setup(seed,steps); generator=torch.Generator().manual_seed(seed+2025)
    theta=torch.zeros(3,requires_grad=True); opt=torch.optim.SGD([theta],lr=.03)
    rows=[]; ticks=grid(steps)
    log(rows,0,evaluate_initialization(theta),emit)
    for t in range(1,steps+1):
        losses=[objective(theta,*task(generator),adapt=False) for _ in range(4)]
        outer=torch.stack(losses).mean()
        opt.zero_grad(); outer.backward(); opt.step()
        if t in ticks:
            log(rows,t,evaluate_initialization(theta),emit,outer_loss=float(outer.detach()),
                training_tasks=4*t,training_examples=64*t,outer_optimizer_steps=t)
    return rows

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META,run)
