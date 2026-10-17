"""后继特征 GPI。
单状态两动作继续任务，γ=0.8，行为动作均匀采样；特征为动作one-hot，SF全0。基础策略库总动作0/总动作1。中点奖励权重[0.2,1]变[1,0.2]并直接告知算法；此实验展示已知线性奖励的迁移，不包含奖励权重学习。
"""
import random
import math
from pathlib import Path
try:
    from ._common import *
except ImportError:
    from _common import *

META = {
    "id": "successor_features_gpi",
    "name": "后继特征 GPI",
    "family": "continual",
    "chapter_paths": [
        "construction/models/",
        "construction/goals/",
        "construction/predictive-knowledge/"
    ],
    "description": "单状态两动作继续任务，γ=0.8，行为动作均匀采样；特征为动作one-hot，SF全0。基础策略库总动作0/总动作1。中点奖励权重[0.2,1]变[1,0.2]并直接告知算法；此实验展示已知线性奖励的迁移，不包含奖励权重学习。",
    "question": "已知奖励权重切换时，GPI怎样组合后继特征？",
    "task": "successor_feature_transfer",
    "baseline": "successor_features_policy",
    "metric": "冻结选择策略的解析折扣价值",
    "unit": "return",
    "higher_better": True,
    "scope": "teaching-control",
    "sources": [
        "http://incompleteideas.net/book/the-book-2nd.html",
        "https://arxiv.org/abs/1606.05312"
    ],
    "defaults": {
        "steps": 1000
    },
    "budget": "environment_steps",
    "budget_note": "每步一个真实转移；中点切换并告知奖励权重；不包含奖励权重学习。",
    "limitations": "小型教学任务，非论文基准复现；不声称持续学习算法的一般有效性。"
}

def update(psi,a,alpha=.1,gamma=.8):
    # 一个状态、两个基础策略π0/π1，每个策略各有两动作的二维SF。
    # 真实转移特征为动作one-hot；所有通道读取旧SF以避免顺序耦合。
    old=[[v[:] for v in policy] for policy in psi]
    for policy in range(2):
        for j in range(2):
            target=float(a==j)+gamma*old[policy][policy][j]
            psi[policy][a][j]+=alpha*(target-old[policy][a][j])
    return psi

def select(psi,reward_weights):
    # 对每个动作先取库中最优基础策略价值，再在动作间最大化。
    scores=[max(dot(psi[p][a],reward_weights) for p in range(2)) for a in range(2)]
    return max(range(2),key=lambda a:scores[a])

def run(seed=0,steps=1000,emit=None):
    check_steps(steps)
    rng,rows=random.Random(seed),[]
    psi=[[[0.,0.],[0.,0.]],[[0.,0.],[0.,0.]]]
    for t in range(1,steps+1):
        a=rng.randrange(2)
        update(psi,a)
        reward_weights=[.2,1.] if t<=steps//2 else [1.,.2]
        chosen=select(psi,reward_weights)
        # 一状态确定策略解析值；评估真值只用于测量。
        log(rows,t,reward_weights[chosen]/(1-.8),steps,emit,selected_action=chosen)
    return rows

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from runtime import run_cli
    run_cli(META, run)
