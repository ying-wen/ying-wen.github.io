"""Ordinary IS
固定三步回合，μ=(0.5,0.5)、π=(0.2,0.8)，动作Bernoulli均值0.1/0.9，γ=0.9。每预算单位是一条真实三转移回合；目标策略固定、比率精确给定，累计估计初始0；不训练控制策略。
W=Πρ_t，估计=ΣWG/N；正确行为概率和支持条件不可省略。
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
    "id": "ordinary_is",
    "name": "Ordinary IS",
    "family": "extended_classic",
    "coverage_refs": [
        "core-ordinary-is"
    ],
    "implementation_kind": "component_experiment",
    "task": "extended_ope_horizon3",
    "baseline": "per_decision_is",
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
    "question": "全轨迹重要性乘积怎样估计固定目标策略回报？",
    "limitations": "仅实现明确列出的子机制；不是原始方法全部模块或论文基准复现。"
}

def importance_ratio(target_probability,behavior_probability):
    if not 0<=target_probability<=1 or not 0<behavior_probability<=1:
        raise ValueError("采样动作需要正行为概率且概率须在合法范围")
    return target_probability/behavior_probability

def sample_return(rewards,ratios,gamma=.9):
    if len(rewards)!=len(ratios) or any(r<0 or not math.isfinite(r) for r in ratios):
        raise ValueError("重要性权重须有限非负且与奖励长度匹配")
    weight=math.prod(ratios)
    discounted=sum(gamma**k*r for k,r in enumerate(rewards))
    return weight*discounted,weight

def update(total,mass,count,rewards,ratios):
    contribution,weight=sample_return(rewards,ratios)
    total+=contribution
    mass+=weight
    count+=1
    # weighted IS有限样本有偏；全部权重为0时返回0的明确约定。
    estimate=total/count
    return total,mass,count,estimate

def run(seed=0,steps=1200,emit=None):
    check(steps)
    rng,rows=random.Random(seed),[]
    total,mass,count,estimate=0.,0.,0,0.
    truth=.74*(1+.9+.9**2)
    for t in range(1,steps+1):
        actions,rewards,ratios=ope_episode(rng)
        total,mass,count,estimate=update(total,mass,count,rewards,ratios)
        log(rows,t,abs(estimate-truth),steps,emit,estimate=estimate,environment_steps=3*t)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
