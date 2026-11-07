"""线性V与softmax策略的完整在线AC(lambda)环境主循环；U=V+.5 logπ。
同任务、同参数/梯度U/采样/迹的固定步长对照，不加meta更新。
真实终止执行当前迹更新后清z/z_beta/u；h/beta/v跨回合保留。
gamma=1，lambda=.8；冻结贪心评价不等同训练的随机策略目标。
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
    "id": "extended-trace_ac",
    "name": "固定步长AC(lambda)对照",
    "family": "extended_adaptation",
    "task": "linear-trace-chain",
    "baseline": "extended-metatrace",
    "metric": "frozen_greedy_return",
    "unit": "return",
    "higher_better": True,
    "scope": "teaching-control",
    "implementation_kind": "teaching_method",
    "coverage_refs": [],
    "chapter_paths": [
        "algorithms/meta/"
    ],
    "description": "固定步长AC(lambda)对照的实际CPU教学实验；设置、公式和边界详见本文件。",
    "question": "实际学习或状态更新怎样改变此教学任务的独立测量？",
    "sources": [
        "https://arxiv.org/html/1805.04514v2"
    ],
    "defaults": {
        "steps": 600
    },
    "budget": "environment_steps",
    "budget_note": "按实际环境转移计步；各算法更新/回放频率见本文件循环，环境步相同不等于相同计算成本。",
    "limitations": "小型教学任务，不是原论文性能复现；组件实验不计作完整原方法。冻结评价与训练更新分离，预算不代表相同计算成本。"
}

def step(weights,trace,g_u,delta,alpha=.03,q=.8):
    trace=q*trace+g_u
    return weights+alpha*delta*trace,trace

def run(seed=0,steps=600,emit=None):
    rng=setup(seed,steps); weights=torch.zeros(18); z=torch.zeros(18); h=torch.zeros(18)
    z_beta=normalizer=bound=0.; beta=math.log(.03); alpha=.03
    env=Chain(); x=env.reset(); rows=[]; ticks=grid(steps)
    log(rows,0,chain_score(ac_actor(weights)),emit)
    for t in range(1,steps+1):
        probs=ac_actor(weights)(x).softmax(-1)
        # Python训练随机流与确定性评价隔离。
        a=int(rng.random()<float(probs[1]))
        xp,r,d=env.step(a)
        delta,g_u,g_delta=ac_terms(weights,x,a,r,xp,d)
        weights,z=step(weights,z,g_u,delta)
        if d:
            z.zero_(); z_beta=bound=0.
        x=env.reset() if d else xp
        if t in ticks:
            log(rows,t,chain_score(ac_actor(weights)),emit,step_size=alpha,
                td_error=delta,trace_norm=float(z.norm()),meta_sensitivity_norm=float(h.norm()))
    return rows

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META,run)
