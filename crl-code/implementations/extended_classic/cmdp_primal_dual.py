"""单状态 CMDP 随机原始对偶
单状态两动作reward Bernoulli均值0.2/0.9，cost=动作1指示，预算0.4，独立真实样本。π1=sigmoid(logit)，初值logit=0、乘子0；解析可行最优π1=0.4只用于测量。对照忽略约束，不是等可行解竞争；不实现一般多状态CMDP占据测度求解。
θ+=α(r−νc)(a−π1)，ν←max(0,ν+α(c−budget))；两个方向使用旧参数。
时序、边界与未覆盖范围见核心注释及 docs/extended-classic.md。
"""
import math
import random
from pathlib import Path
try:
    from ._common import *
except ImportError:
    from _common import *

META = {
    "id": "cmdp_primal_dual",
    "name": "单状态 CMDP 随机原始对偶",
    "family": "extended_classic",
    "coverage_refs": [
        "core-primal-dual"
    ],
    "implementation_kind": "component_experiment",
    "task": "extended_constrained_bandit",
    "baseline": "unconstrained_policy_reference",
    "metric": "最优可行动作概率绝对误差",
    "unit": "probability",
    "higher_better": False,
    "budget": "environment_steps",
    "scope": "component-constrained-policy-update",
    "chapter_paths": [
        "foundations/deep/constraints/"
    ],
    "sources": [
        "https://arxiv.org/abs/2101.10895"
    ],
    "defaults": {
        "steps": 1200
    },
    "description": "单状态两动作reward Bernoulli均值0.2/0.9，cost=动作1指示，预算0.4，独立真实样本。π1=sigmoid(logit)，初值logit=0、乘子0；解析可行最优π1=0.4只用于测量。对照忽略约束，不是等可行解竞争；不实现一般多状态CMDP占据测度求解。",
    "question": "随机原始对偶方向怎样响应单状态代价约束？",
    "limitations": "仅实现明确列出的子机制；不是原始方法全部模块或论文基准复现。"
}

def update(logit,multiplier,action,reward,cost,t,budget=.4):
    p=sigmoid(logit)
    # r-νc的score-function梯度；ν旧值同时驱动actor和dual。
    rate=.15/math.sqrt(t)
    actor_direction=(reward-multiplier*cost)*(action-p)
    new_logit=logit+rate*actor_direction
    new_multiplier=max(0.,multiplier+rate*(cost-budget))
    return new_logit,new_multiplier

def run(seed=0,steps=1200,emit=None):
    check(steps)
    rng,rows,logit,multiplier=random.Random(seed),[],0.,0.
    for t in range(1,steps+1):
        p=sigmoid(logit)
        a=int(rng.random()<p)
        reward=float(rng.random()<[.2,.9][a])
        cost=float(a)
        logit,multiplier=update(logit,multiplier,a,reward,cost,t)
        p=sigmoid(logit)
        log(rows,t,abs(p-.4),steps,emit,cost_violation=max(0.,p-.4),expected_reward=.2+.7*p,multiplier=multiplier)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
