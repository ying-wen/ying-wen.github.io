"""无探索奖励 Q-learning。
平稳六格链从0开始，终点奖励1、其余-0.01，γ=0.95，ε=0.1，Q=0。状态动作计数从0开始，每次访问先加1，再计算0.2/√N探索奖励；对照系数0。冻结测量原始奖励贪心回报；计数新奇不等于信息增益。
"""
import random
import math
from pathlib import Path
try:
    from ._common import *
except ImportError:
    from _common import *

META = {
    "id": "plain_q_exploration",
    "name": "无探索奖励 Q-learning",
    "family": "continual",
    "chapter_paths": [
        "algorithms/exploration/",
        "foundations/reward-design/",
        "algorithms/control/"
    ],
    "description": "平稳六格链从0开始，终点奖励1、其余-0.01，γ=0.95，ε=0.1，Q=0。状态动作计数从0开始，每次访问先加1，再计算0.2/√N探索奖励；对照系数0。冻结测量原始奖励贪心回报；计数新奇不等于信息增益。",
    "question": "访问计数探索奖励怎样影响平稳链的原奖励回报？",
    "task": "count_exploration_chain",
    "baseline": "count_bonus",
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
    "budget_note": "每步一个原始转移；任务平稳，无中点变化；训练加bonus，评价原奖励。",
    "limitations": "小型教学任务，非论文基准复现；不声称持续学习算法的一般有效性。"
}

def update(q,s,a,reward,sp,count,alpha=.2,beta=0.):
    # 奖励仅在训练目标添加；评价始终使用原环境奖励。
    bonus=beta/math.sqrt(count)
    target=reward+bonus+(.95*max(q[sp]) if sp is not None else 0.)
    q[s][a]+=alpha*(target-q[s][a])
    return bonus

def run(seed=0,steps=1000,emit=None):
    check_steps(steps)
    rng,rows,q,s=random.Random(seed),[],{j:[0.,0.] for j in range(5)},0
    counts={j:[0,0] for j in range(5)}
    for t in range(1,steps+1):
        a=eps_action(q[s],rng)
        reward,sp=chain(s,a)
        counts[s][a]+=1
        bonus=update(q,s,a,reward,sp,counts[s][a])
        s=0 if sp is None else sp
        log(rows,t,chain_score(q),steps,emit,phase="stationary",exploration_bonus=bonus)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
