"""首次访问 MC control
平稳六格链从0开始，终点奖励1、其余-0.01，γ=0.95，ε=0.1。初始估值0，每预算单位是真实环境转移；冻结贪心策略用终点/循环精确原奖励回报评价。MC每回合固定当前ε-soft策略，只在完整回合首次状态动作访问更新。
Q(s,a)←Q(s,a)+(G_t−Q(s,a))/N(s,a)；回合结束后才变化策略。
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
    "id": "mc_control",
    "name": "首次访问 MC control",
    "family": "extended_classic",
    "coverage_refs": [
        "core-mc-control"
    ],
    "implementation_kind": "teaching_method",
    "task": "extended_chain_control",
    "baseline": "watkins_q_lambda",
    "metric": "贪心策略精确折扣回报",
    "unit": "return",
    "higher_better": True,
    "budget": "environment_steps",
    "scope": "teaching-control",
    "chapter_paths": [
        "foundations/tabular/monte-carlo/",
        "algorithms/control/"
    ],
    "sources": [
        "http://incompleteideas.net/book/the-book-2nd.html"
    ],
    "defaults": {
        "steps": 1200
    },
    "description": "平稳六格链从0开始，终点奖励1、其余-0.01，γ=0.95，ε=0.1。初始估值0，每预算单位是真实环境转移；冻结贪心策略用终点/循环精确原奖励回报评价。MC每回合固定当前ε-soft策略，只在完整回合首次状态动作访问更新。",
    "question": "首次状态动作回报怎样改善ε-soft控制？",
    "limitations": "有限教学任务，不构成一般性能或原论文实验复现。"
}

def update(q,counts,trajectory):
    g,targets=0.,[0.]*len(trajectory)
    for k in reversed(range(len(trajectory))):
        g=trajectory[k][2]+.95*g
        targets[k]=g
    seen=set()
    for (s,a,_),target in zip(trajectory,targets):
        if (s,a) not in seen:
            seen.add((s,a))
            counts[s][a]+=1
            q[s][a]+=(target-q[s][a])/counts[s][a]

def run(seed=0,steps=1200,emit=None):
    check(steps)
    rng,rows,q,s=random.Random(seed),[],{j:[0.,0.] for j in range(5)},0
    counts={j:[0,0] for j in range(5)}
    trajectory=[]
    for t in range(1,steps+1):
        a=sample(probs(q[s]),rng)
        r,sp=chain(s,a)
        trajectory.append((s,a,r))
        if sp is None:
            update(q,counts,trajectory)
            trajectory=[]
        s=0 if sp is None else sp
        log(rows,t,score(q),steps,emit)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
