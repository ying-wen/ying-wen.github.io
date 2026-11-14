"""链任务 Q-learning。
六格链从0开始，终点奖励1、其余-0.01，γ=0.95，ε=0.1，Q=0。势函数固定、在线更新后冻结贪心策略，以原始环境奖励评价。任务全程平稳，无中点切换；塑形只改变训练奖励。
"""
import random
import math
from pathlib import Path
try:
    from ._common import *
except ImportError:
    from _common import *

META = {
    "id": "chain_q",
    "name": "链任务 Q-learning",
    "family": "continual",
    "chapter_paths": [
        "foundations/reward-design/",
        "algorithms/control/",
        "foundations/objectives/"
    ],
    "description": "六格链从0开始，终点奖励1、其余-0.01，γ=0.95，ε=0.1，Q=0。势函数固定、在线更新后冻结贪心策略，以原始环境奖励评价。任务全程平稳，无中点切换；塑形只改变训练奖励。",
    "question": "固定势函数怎样改变平稳链的原奖励学习回报？",
    "task": "potential_chain_control",
    "baseline": "potential_shaping",
    "metric": "原始奖励贪心策略折扣回报",
    "unit": "return",
    "higher_better": True,
    "scope": "teaching-control",
    "sources": [
        "http://incompleteideas.net/book/the-book-2nd.html"
    ],
    "defaults": {
        "steps": 1000
    },
    "budget": "environment_steps",
    "budget_note": "每步一个原始转移；任务平稳，无中点变化；冻结策略精确测量终点或无限循环折扣回报。",
    "limitations": "小型教学任务，非论文基准复现；不声称持续学习算法的一般有效性。"
}

def update(q,s,a,reward,sp,alpha=.2):
    target=reward+(.95*max(q[sp]) if sp is not None else 0.)
    q[s][a]+=alpha*(target-q[s][a])
    return target

def run(seed=0,steps=1000,emit=None):
    check_steps(steps)
    rng,rows,q,s=random.Random(seed),[],{j:[0.,0.] for j in range(5)},0
    for t in range(1,steps+1):
        a=eps_action(q[s],rng)
        reward,sp=chain(s,a)
        training_reward=reward
        update(q,s,a,training_reward,sp)
        s=0 if sp is None else sp
        log(rows,t,chain_score(q),steps,emit,phase="stationary")
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
