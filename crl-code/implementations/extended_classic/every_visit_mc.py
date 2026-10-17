"""Every-visit MC
五状态无偏随机游走，从3开始，终点左右奖励0/1，γ=1，价值初始0。仅完整回合产生MC更新；预算尾未完成回合不伪终止。
G_t=Σ未来奖励；V(s)←V(s)+(G_t−V(s))/N(s)，每次访问都计数。
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
    "id": "every_visit_mc",
    "name": "Every-visit MC",
    "family": "extended_classic",
    "coverage_refs": [
        "core-every-visit-mc"
    ],
    "implementation_kind": "teaching_method",
    "task": "extended_walk_mc",
    "baseline": "first_visit_mc_reference",
    "metric": "五状态价值 RMSE",
    "unit": "value",
    "higher_better": False,
    "budget": "environment_steps",
    "scope": "teaching-prediction",
    "chapter_paths": [
        "foundations/tabular/monte-carlo/",
        "algorithms/value/"
    ],
    "sources": [
        "http://incompleteideas.net/book/the-book-2nd.html"
    ],
    "defaults": {
        "steps": 1200
    },
    "description": "五状态无偏随机游走，从3开始，终点左右奖励0/1，γ=1，价值初始0。仅完整回合产生MC更新；预算尾未完成回合不伪终止。",
    "question": "每次访问与首次访问怎样改变同回合的MC样本计数？",
    "limitations": "有限教学任务，不构成一般性能或原论文实验复现。"
}

def update(v,counts,states,rewards):
    targets,g=[0.]*len(rewards),0.
    for k in reversed(range(len(rewards))):
        g=rewards[k]+g
        targets[k]=g
    seen=set()
    for s,target in zip(states,targets):
        # 同一回合再次访问仍计入样本；各回报相关不影响均值定义。
        seen.add(s)
        counts[s]+=1
        v[s]+=(target-v[s])/counts[s]

def run(seed=0,steps=1200,emit=None):
    check(steps)
    rng,rows=random.Random(seed),[]
    v,counts,s=dict.fromkeys(range(1,6),0.),dict.fromkeys(range(1,6),0),3
    states,rewards=[],[]
    for t in range(1,steps+1):
        r,sp=walk(s,rng)
        states.append(s)
        rewards.append(r)
        if sp is None:
            update(v,counts,states,rewards)
            states,rewards=[],[]
        s=3 if sp is None else sp
        log(rows,t,value_error(v),steps,emit)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
