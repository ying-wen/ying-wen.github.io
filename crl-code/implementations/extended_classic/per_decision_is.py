"""Per-decision IS
固定三步回合，μ=(0.5,0.5)、π=(0.2,0.8)，动作Bernoulli均值0.1/0.9，γ=0.9。每预算单位是一条真实三转移回合；目标策略固定、比率精确给定，累计估计初始0；不训练控制策略。
每个奖励仅乘到达其时刻前的ρ乘积：Σγ^t(Πi≤tρ_i)r_t。
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
    "id": "per_decision_is",
    "name": "Per-decision IS",
    "family": "extended_classic",
    "coverage_refs": [
        "core-pdis"
    ],
    "implementation_kind": "component_experiment",
    "task": "extended_ope_horizon3",
    "baseline": "ordinary_is",
    "metric": "目标策略回报估计绝对误差",
    "unit": "value",
    "higher_better": False,
    "budget": "episodes",
    "scope": "component-off-policy-evaluation",
    "chapter_paths": [
        "foundations/deep/offline/",
        "foundations/tabular/monte-carlo/"
    ],
    "sources": [
        "https://arxiv.org/abs/1511.03722",
        "https://web.eecs.umich.edu/~baveja/Papers/OffPolicy.pdf"
    ],
    "defaults": {
        "steps": 1200
    },
    "description": "固定三步回合，μ=(0.5,0.5)、π=(0.2,0.8)，动作Bernoulli均值0.1/0.9，γ=0.9。每预算单位是一条真实三转移回合；目标策略固定、比率精确给定，累计估计初始0；不训练控制策略。",
    "question": "逐决策权重怎样减少不必要的未来动作乘积？",
    "limitations": "仅实现明确列出的子机制；不是原始方法全部模块或论文基准复现。"
}

def importance_ratio(target_probability,behavior_probability):
    if not 0<=target_probability<=1 or not 0<behavior_probability<=1:
        raise ValueError("采样动作需要正行为概率且概率须在合法范围")
    return target_probability/behavior_probability

def sample_return(rewards,ratios,gamma=.9):
    if len(rewards)!=len(ratios) or any(r<0 or not math.isfinite(r) for r in ratios):
        raise ValueError("重要性权重须有限非负且与奖励长度匹配")
    result,weight=0.,1.
    for k,(r,rho) in enumerate(zip(rewards,ratios)):
        weight*=rho
        result+=gamma**k*weight*r
    return result

def run(seed=0,steps=1200,emit=None):
    check(steps)
    rng,rows=random.Random(seed),[]
    total,mass,count,estimate=0.,0.,0,0.
    truth=.74*(1+.9+.9**2)
    for t in range(1,steps+1):
        actions,rewards,ratios=ope_episode(rng)
        total+=sample_return(rewards,ratios)
        estimate=total/t
        log(rows,t,abs(estimate-truth),steps,emit,estimate=estimate,environment_steps=3*t)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
