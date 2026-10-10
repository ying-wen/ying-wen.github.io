"""UCB 对照
三臂平稳Bernoulli均值0.2/0.5/0.8，初始无观测，Thompson使用独立Beta(1,1)先验。真实选择和奖励逐步更新，报告实际累计平均奖励，不使用真均值选动作。
未访问先尝试；随后argmax[均值+√(2logt/N)]，只使用已观测奖励。
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
    "id": "ucb_extended_baseline",
    "name": "UCB 对照",
    "family": "extended_classic",
    "coverage_refs": [],
    "implementation_kind": "teaching_method",
    "task": "extended_stationary_bandit",
    "baseline": "thompson_sampling",
    "metric": "累计采样平均奖励",
    "unit": "reward",
    "higher_better": True,
    "budget": "environment_steps",
    "scope": "teaching-bandit-control",
    "chapter_paths": [
        "foundations/deep/exploration/",
        "algorithms/exploration/"
    ],
    "sources": [
        "https://link.springer.com/article/10.1023/A:1013689704352"
    ],
    "defaults": {
        "steps": 1200
    },
    "description": "三臂平稳Bernoulli均值0.2/0.5/0.8，初始无观测，Thompson使用独立Beta(1,1)先验。真实选择和奖励逐步更新，报告实际累计平均奖励，不使用真均值选动作。",
    "question": "UCB均值和置信奖励怎样产生赌博机动作选择？",
    "limitations": "有限教学任务，不构成一般性能或原论文实验复现。"
}

def select(successes,failures,rng,t):
    counts=[s+f for s,f in zip(successes,failures)]
    unseen=[a for a,n in enumerate(counts) if n==0]
    if unseen:
        return unseen[0]
    return max(range(3),key=lambda a:successes[a]/counts[a]+math.sqrt(2*math.log(t)/counts[a]))

def update(successes,failures,a,reward):
    successes[a]+=int(reward)
    failures[a]+=int(1-reward)

def run(seed=0,steps=1200,emit=None):
    check(steps)
    rng,rows,successes,failures,total=random.Random(seed),[],[0]*3,[0]*3,0.
    for t in range(1,steps+1):
        a=select(successes,failures,rng,t)
        reward=float(rng.random()<[.2,.5,.8][a])
        update(successes,failures,a,reward)
        total+=reward
        log(rows,t,total/t,steps,emit)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
