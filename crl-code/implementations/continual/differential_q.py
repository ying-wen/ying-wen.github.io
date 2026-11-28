"""差分 Q-learning。
单状态无终点两动作继续任务，中点均值[0.2,0.8]交换；Bernoulli奖励。Q=0、奖励率=0，ε=0.1。冻结测量当前贪心动作真实平均奖励，训练不使用真均值。
"""
import random
import math
from pathlib import Path
try:
    from ._common import *
except ImportError:
    from _common import *

META = {
    "id": "differential_q",
    "name": "差分 Q-learning",
    "family": "continual",
    "chapter_paths": [
        "algorithms/average-reward/",
        "algorithms/control/",
        "foundations/objectives/"
    ],
    "description": "单状态无终点两动作继续任务，中点均值[0.2,0.8]交换；Bernoulli奖励。Q=0、奖励率=0，ε=0.1。冻结测量当前贪心动作真实平均奖励，训练不使用真均值。",
    "question": "差分价值和奖励率怎样响应动作均值交换？",
    "task": "continuing_bandit_control",
    "baseline": "differential_sarsa",
    "metric": "贪心动作真实平均奖励",
    "unit": "reward",
    "higher_better": True,
    "scope": "teaching-control",
    "sources": [
        "http://incompleteideas.net/book/the-book-2nd.html"
    ],
    "defaults": {
        "steps": 1000
    },
    "budget": "environment_steps",
    "budget_note": "每步一个继续转移；中点动作Bernoulli均值交换，无终点。",
    "limitations": "小型教学任务，非论文基准复现；不声称持续学习算法的一般有效性。"
}

def update(q,rate,a,reward,ap,alpha=.05,eta=.05):
    # 单状态继续任务：两个更新都用同一个旧差分TD误差。
    next_value=max(q)
    delta=reward-rate+next_value-q[a]
    new_q=q[:]
    new_q[a]+=alpha*delta
    return new_q,rate+eta*alpha*delta,delta

def run(seed=0,steps=1000,emit=None):
    check_steps(steps)
    rng,rows,q,rate=random.Random(seed),[],[0.,0.],0.
    a=eps_action(q,rng)
    for t in range(1,steps+1):
        means=[.2,.8] if t<=steps//2 else [.8,.2]
        reward=float(rng.random()<means[a])
        ap=eps_action(q,rng)
        q,rate,delta=update(q,rate,a,reward,ap)
        a=ap
        best=max(range(2),key=lambda j:q[j])
        log(rows,t,means[best],steps,emit,reward_rate=rate,td_error=delta)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
