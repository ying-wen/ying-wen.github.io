"""Watkins Q(λ)
平稳六格链从0开始，终点奖励1、其余-0.01，γ=0.95，ε=0.1。初始估值0，每预算单位是真实环境转移；冻结贪心策略用终点/循环精确原奖励回报评价。累积迹λ=0.8；非贪心下一动作先按旧Q判定，本次误差信用更新后才清除未来资格。
δ=r+γmaxQ'−Q，e(s,a)+=1，Q+=αδe；按旧Q判定下一动作，再衰减或清迹。
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
    "id": "watkins_q_lambda",
    "name": "Watkins Q(λ)",
    "family": "extended_classic",
    "coverage_refs": [
        "core-watkins-q-lambda"
    ],
    "implementation_kind": "teaching_method",
    "task": "extended_chain_control",
    "baseline": "mc_control",
    "metric": "贪心策略精确折扣回报",
    "unit": "return",
    "higher_better": True,
    "budget": "environment_steps",
    "scope": "teaching-control",
    "chapter_paths": [
        "algorithms/credit-assignment/",
        "algorithms/control/"
    ],
    "sources": [
        "https://www.repository.cam.ac.uk/items/f1ce2aa9-2e01-4a08-92b7-1eec76b652ed",
        "https://arxiv.org/abs/1606.02647"
    ],
    "defaults": {
        "steps": 1200
    },
    "description": "平稳六格链从0开始，终点奖励1、其余-0.01，γ=0.95，ε=0.1。初始估值0，每预算单位是真实环境转移；冻结贪心策略用终点/循环精确原奖励回报评价。累积迹λ=0.8；非贪心下一动作先按旧Q判定，本次误差信用更新后才清除未来资格。",
    "question": "非贪心行为怎样改变未来资格而保留当前信用？",
    "limitations": "有限教学任务，不构成一般性能或原论文实验复现。"
}

def update(q,e,s,a,reward,sp,ap,alpha=.15,lam=.8):
    greedy=sp is not None and q[sp][ap]==max(q[sp])
    delta=reward+(.95*max(q[sp]) if sp is not None else 0.)-q[s][a]
    e[s][a]+=1.
    for j in q:
        for b in range(2):
            q[j][b]+=alpha*delta*e[j][b]
    # 下一动作若与最大值并列也保留迹。终点先给当前奖励信用，再清迹。
    for j in e:
        for b in range(2):
            e[j][b]=.95*lam*e[j][b] if greedy else 0.
    return delta

def run(seed=0,steps=1200,emit=None):
    check(steps)
    rng,rows,q,s=random.Random(seed),[],{j:[0.,0.] for j in range(5)},0
    e={j:[0.,0.] for j in range(5)}
    a=sample(probs(q[s]),rng)
    for t in range(1,steps+1):
        r,sp=chain(s,a)
        ap=sample(probs(q[sp]),rng) if sp is not None else 0
        delta=update(q,e,s,a,r,sp,ap)
        s=0 if sp is None else sp
        a=sample(probs(q[s]),rng) if sp is None else ap
        log(rows,t,score(q),steps,emit,td_error=delta)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
