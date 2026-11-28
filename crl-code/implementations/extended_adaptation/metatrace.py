"""线性V与softmax策略的完整在线AC(lambda)环境主循环；U=V+.5 logπ。
忠实实现论文Algorithm1的归一化scalar beta/z_beta/h/u/v时序，entropy系数psi=0。
实际更新critic与policy，不是原critic-only scalar核；但不包含vector/mixed版本和熵项，因此组件实验。
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
    "id": "extended-metatrace",
    "name": "归一化Scalar Metatrace AC组件",
    "family": "extended_adaptation",
    "task": "linear-trace-chain",
    "baseline": "extended-trace_ac",
    "metric": "frozen_greedy_return",
    "unit": "return",
    "higher_better": True,
    "scope": "teaching-control",
    "implementation_kind": "component_experiment",
    "coverage_refs": [
        "core-metatrace"
    ],
    "chapter_paths": [
        "algorithms/meta/"
    ],
    "description": "归一化Scalar Metatrace AC组件的实际CPU教学实验；设置、公式和边界详见本文件。",
    "question": "实际学习或状态更新怎样改变此教学任务的独立测量？",
    "sources": [
        "https://arxiv.org/html/1805.04514v2"
    ],
    "defaults": {
        "steps": 600
    },
    "budget": "environment_steps",
    "budget_note": "按实际环境转移计步；各算法更新/回放频率见本文件循环，环境步相同不等于相同计算成本。",
    "limitations": "normalized scalar Metatrace AC(lambda) 特例；忽略trace参数Jacobian的局部敏感度近似，不是per-parameter变体。冻结贪心回报不等同随机策略目标。"
}

def step(weights,h,z,z_beta,beta,normalizer,bound,g_u,g_delta,delta,mu=.01,q=.8):
    # Algorithm1：z_beta读取旧h，running max归一化beta，再约束effective step。
    z=q*z+g_u
    z_beta=q*z_beta+float(g_u@h)
    beta_delta=z_beta*delta
    normalizer=max(abs(beta_delta),normalizer+mu*(abs(beta_delta)-normalizer))
    beta+=mu*beta_delta/(normalizer if normalizer>0 else 1.)
    effective=math.exp(beta)*float(g_u.square().sum())
    bound=max(effective,bound+(1-q)*(effective-bound))
    beta-=math.log(max(bound,1.))
    alpha=math.exp(beta)
    # ∂z/∂w≈0：非线性actor中h为论文的局部敏感度近似，非全轨迹精确梯度。
    new_h=h+alpha*z*(delta+float(g_delta@h))
    return weights+alpha*delta*z,new_h,z,z_beta,beta,normalizer,bound,alpha

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
        weights,h,z,z_beta,beta,normalizer,bound,alpha=step(weights,h,z,z_beta,beta,normalizer,bound,g_u,g_delta,delta)
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
