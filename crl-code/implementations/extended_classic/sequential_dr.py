"""Sequential Doubly Robust
固定三步回合，μ=(0.5,0.5)、π=(0.2,0.8)，动作Bernoulli均值0.1/0.9，γ=0.9。每预算单位是一条真实三转移回合；目标策略固定、比率精确给定，累计估计初始0；不训练控制策略。
DR_t=Vhat_t+ρ_t[r_t+γDR_(t+1)−Qhat_t]，真终点尾=0。
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
    "id": "sequential_dr",
    "name": "Sequential Doubly Robust",
    "family": "extended_classic",
    "coverage_refs": [
        "core-sequential-dr"
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
    "question": "冻结旧模型怎样与准确比率形成序贯DR估计？",
    "limitations": "仅实现明确列出的子机制；不是原始方法全部模块或论文基准复现。"
}

def importance_ratio(target_probability,behavior_probability):
    if not 0<=target_probability<=1 or not 0<behavior_probability<=1:
        raise ValueError("采样动作需要正行为概率且概率须在合法范围")
    return target_probability/behavior_probability

def sample_return(rewards,ratios,q_hat,v_hat,gamma=.9):
    if len(rewards)!=len(ratios) or any(r<0 or not math.isfinite(r) for r in ratios):
        raise ValueError("重要性权重须有限非负且与奖励长度匹配")
    if len(q_hat)!=len(rewards) or len(v_hat)!=len(rewards):
        raise ValueError("模型动作价值和状态价值需与真实奖励对齐")
    # 反向序贯DR：模型预测冻结在本条episode开始之前。
    # Vhat与Qhat必须来自同一模型和目标π；终点Vhat=0。
    value=0.
    for k in reversed(range(len(rewards))):
        value=v_hat[k]+ratios[k]*(rewards[k]+gamma*value-q_hat[k])
    return value

def run(seed=0,steps=1200,emit=None):
    check(steps)
    rng,rows=random.Random(seed),[]
    total,mass,count,estimate=0.,0.,0,0.
    counts=[[0,0] for _ in range(3)]
    sums=[[0.,0.] for _ in range(3)]
    truth=.74*(1+.9+.9**2)
    for t in range(1,steps+1):
        actions,rewards,ratios=ope_episode(rng)
        means=[[sums[k][a]/counts[k][a] if counts[k][a] else .4 for a in range(2)] for k in range(3)]
        vhat=[0.]*4
        for k in reversed(range(3)):
            vhat[k]=dot([.2,.8],means[k])+.9*vhat[k+1]
        qhat=[means[k][a]+.9*vhat[k+1] for k,a in enumerate(actions)]
        total+=sample_return(rewards,ratios,qhat,vhat[:-1])
        estimate=total/t
        # 预测器拟合在DR估计完成后才吸收当前样本，避免同样本泄漏。
        for k,a in enumerate(actions):
            counts[k][a]+=1
            sums[k][a]+=rewards[k]
        log(rows,t,abs(estimate-truth),steps,emit,estimate=estimate,environment_steps=3*t)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
